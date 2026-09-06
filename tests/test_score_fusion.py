"""Tests de la Fase 3 (src/engines/score_fusion.py): fusion continua de scores y el
mecanismo de completado hasta ``TARGET_CANDIDATES`` + reserva perezosa.

Logica pura (sin subprocess), se prueba de punta a punta con DataFrames sinteticos
de scores crudos por residuo (2 motores, combinacion bepipred+epidope salvo que se
indique otra), controlando el score normalizado de cada residuo para producir
patrones de umbral predecibles.
"""

import pandas as pd
import pytest

from src.engines import bepipred_engine, discotope_engine, epidope_engine, scannet_engine
from src.engines.score_fusion import (
    FALLBACK_WINDOW,
    FUSION_BOUNDS,
    RESERVE_EXTRA,
    TARGET_CANDIDATES,
    fuse_and_extract_regions,
)

ACCESSION = "TEST_PROT"


def _raw_for_norm(engine: str, norm: float) -> float:
    """Score crudo que, tras normalizar con ``FUSION_BOUNDS[engine]``, da ``norm`` exacto."""
    lo, hi = FUSION_BOUNDS[engine]
    return lo + norm * (hi - lo)


def _engine_df(engine: str, norms):
    accession_col = {"bepipred": bepipred_engine.ACCESSION_COLUMN, "epidope": epidope_engine.ACCESSION_COLUMN,
                      "discotope": discotope_engine.ACCESSION_COLUMN, "scannet": scannet_engine.ACCESSION_COLUMN}[engine]
    score_col = {"bepipred": bepipred_engine.SCORE_COLUMN, "epidope": epidope_engine.SCORE_COLUMN,
                 "discotope": discotope_engine.SCORE_COLUMN, "scannet": scannet_engine.SCORE_COLUMN}[engine]
    residue_col = {"bepipred": "Residue", "epidope": epidope_engine.RESIDUE_COLUMN,
                   "discotope": discotope_engine.RESIDUE_COLUMN, "scannet": scannet_engine.RESIDUE_COLUMN}[engine]
    return pd.DataFrame({
        accession_col: [ACCESSION] * len(norms),
        residue_col: ["A"] * len(norms),
        score_col: [_raw_for_norm(engine, n) for n in norms],
    })


def _fuse_two_engines(engine_a: str, engine_b: str, norms_a, norms_b):
    raw_dfs = {engine_a: _engine_df(engine_a, norms_a), engine_b: _engine_df(engine_b, norms_b)}
    sequence_lookup = {ACCESSION: "A" * len(norms_a)}
    return fuse_and_extract_regions(raw_dfs, sequence_lookup)


def _regions(df: pd.DataFrame):
    """Lista de (start, end) 1-indexados, para chequear solape entre filas del resultado."""
    return list(zip(df["start"], df["end"]))


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _assert_no_overlaps(df: pd.DataFrame):
    regions = _regions(df)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            assert not _overlaps(regions[i], regions[j]), f"regiones solapadas: {regions[i]} y {regions[j]}"


class TestSinCompletado:
    """Antigenos que YA alcanzan TARGET_CANDIDATES por umbral: sin top_up ni reserve."""

    def test_tres_regiones_sobre_umbral_no_generan_completado(self):
        n = 40
        norms_a = [0.0] * n
        norms_b = [0.0] * n
        for start in (0, 15, 30):
            for i in range(start, start + 9):
                norms_a[i] = 1.0
                norms_b[i] = 1.0
        df = _fuse_two_engines("bepipred", "epidope", norms_a, norms_b)

        assert len(df) == 3
        assert set(df["candidate_source"]) == {"threshold"}
        assert not df["fallback"].any()
        _assert_no_overlaps(df)


class TestCompletadoYReserva:
    """Antigenos con menos de TARGET_CANDIDATES regiones sobre umbral: se completa + reserva."""

    def test_una_region_sobre_umbral_completa_hasta_target_y_anade_reserva(self):
        n = 100
        norms_a = [0.3] * n
        norms_b = [0.3] * n  # fused=0.6 en todas partes, por debajo de 1.164683
        for i in range(0, 9):
            norms_a[i] = 1.0
            norms_b[i] = 1.0  # fused=2.0, unica region sobre umbral
        df = _fuse_two_engines("bepipred", "epidope", norms_a, norms_b)

        assert len(df) == TARGET_CANDIDATES + RESERVE_EXTRA  # 1 threshold + 2 top_up + 3 reserve
        counts = df["candidate_source"].value_counts()
        assert counts.get("threshold", 0) == 1
        assert counts.get("top_up", 0) == TARGET_CANDIDATES - 1
        assert counts.get("reserve", 0) == RESERVE_EXTRA
        assert df.loc[df["candidate_source"] == "threshold", "fallback"].eq(False).all()
        assert df.loc[df["candidate_source"] != "threshold", "fallback"].eq(True).all()
        assert (df["length"] == FALLBACK_WINDOW).all()
        _assert_no_overlaps(df)

    def test_cero_regiones_sobre_umbral_con_espacio_da_target_mas_reserva(self):
        n = 100
        norms_a = [0.1] * n
        norms_b = [0.1] * n  # fused=0.2 en todas partes, ninguna region sobre umbral
        df = _fuse_two_engines("bepipred", "epidope", norms_a, norms_b)

        assert len(df) == TARGET_CANDIDATES + RESERVE_EXTRA
        counts = df["candidate_source"].value_counts()
        assert counts.get("threshold", 0) == 0
        assert counts.get("top_up", 0) == TARGET_CANDIDATES
        assert counts.get("reserve", 0) == RESERVE_EXTRA
        assert df["fallback"].all()
        _assert_no_overlaps(df)

    def test_proteina_corta_sin_espacio_para_segunda_ventana_da_un_unico_candidato(self):
        # 13 residuos (como 8FDD/6B5M): 2 ventanas de 9 no caben sin solape.
        n = 13
        norms_a = [0.1] * n
        norms_b = [0.1] * n
        df = _fuse_two_engines("bepipred", "epidope", norms_a, norms_b)

        assert len(df) == 1
        assert df.iloc[0]["candidate_source"] == "top_up"
        assert bool(df.iloc[0]["fallback"]) is True
        assert df.iloc[0]["length"] == FALLBACK_WINDOW  # 9 de 13 residuos; no cabe una segunda ventana sin solape


class TestCombinacionesDeMotores:
    """Las 3 combinaciones de motores soportadas siguen resolviendo un umbral valido."""

    def test_combinacion_discotope_scannet(self):
        n = 30
        norms_dt = [0.0] * n
        norms_sn = [0.0] * n
        for i in range(0, 9):
            norms_dt[i] = 1.0
            norms_sn[i] = 1.0
        df = _fuse_two_engines("discotope", "scannet", norms_dt, norms_sn)
        assert not df.empty
        assert "discotope_score" in df.columns and "scannet_score" in df.columns

    def test_combinacion_no_calibrada_lanza_value_error(self):
        raw_dfs = {"bepipred": _engine_df("bepipred", [0.5] * 20)}
        with pytest.raises(ValueError):
            fuse_and_extract_regions(raw_dfs, {ACCESSION: "A" * 20})
