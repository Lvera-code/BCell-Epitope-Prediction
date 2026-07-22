"""Fase 7: ensamblaje automatico del constructo multi-epitopo vacunal.

Toma los candidatos que ya sobrevivieron el resto del pipeline (B-cell de
Fase 4/4b/4c, HTL de Fase 5, CTL de Fase 5b), selecciona los mejores
``Settings.CONSTRUCT_TOP_N_PER_CLASS`` por clase, y los concatena con los
linkers estandar del campo de diseno de vacunas multi-epitopo en un unico
FASTA, junto con una tabla de metadata 100% trazable (que peptido individual
aporto cada tramo, en que orden, que linker se uso en cada union).

Reglas de ensamblaje (decision FINAL del usuario, 2026-07-22, convencion
estandar del campo con multiples fuentes que coinciden):

* Linker intra-bloque CTL: ``AAY`` -- sitio de corte del proteasoma en
  celulas de mamifero, libera cada epitopo correctamente durante el
  procesamiento antigenico y evita epitopos de union espurios en la union.
* Linker intra-bloque HTL: ``GPGPG`` -- espaciador universal (Livingston
  et al. 2002), rompe la inmunogenicidad de union y preserva la capacidad
  de inducir respuesta Th de cada epitopo.
* Linker intra-bloque B-cell: ``KK`` -- mantiene la especificidad de cada
  epitopo individual.
* Linker entre bloques de distinta clase: ``GPGPG`` (el mismo espaciador
  universal, usado consistentemente en la literatura como puente entre
  bloques, no solo intra-HTL).
* Orden de bloques: B-cell -> HTL -> CTL. Decision final: sin consenso
  fuerte en la literatura sobre el orden optimo (los linkers ya garantizan
  liberacion correcta por procesamiento antigenico, independiente de la
  posicion); se ancla en B-cell por ser el foco humoral original del
  proyecto (bnAb/HIV).
* Sin adjuvante: decision ACTIVA de no incluir uno en esta version -- la
  eleccion (beta-defensina, PADRE, flagelina, L7/L12, etc.) requiere
  criterio biologico/estrategico especifico del patogeno/huesped, fuera de
  scope de este pipeline. ``assemble_construct`` acepta un
  ``adjuvant_sequence`` opcional (``None`` por defecto) para insertarlo en
  el N-terminal con su propio linker rigido EAAAK (Arai et al. 2001) sin
  rediseñar nada, si se decide agregar uno mas adelante.

Seleccion top-N por clase (decision del usuario, 2026-07-22, confirmada tras
mostrar numeros reales de una corrida con GP120: Fase 5/HTL dio 18
candidatos validos, demasiados para un constructo manejable):

* B-cell: de ``safe_df`` (Fase 4 'Segura'), se descartan los marcados
  'Allergen' por AlgPred2 (Fase 4b) y los que tienen AL MENOS un sequon
  marcado 'Glicosilado' por StackGlyEmbed (Fase 4c) -- un peptido SIN
  ningun sequon nunca aparece en el reporte de Fase 4c y se trata como
  "sin riesgo" (no es lo mismo que "evaluado y limpio", pero es la unica
  lectura consistente: Fase 4c solo produce filas para sequones reales).
  De los que sobreviven, top-N por el MAYOR de sus ``'{motor}_score'``
  disponibles (BepiPred/EpiDope/DiscoTope/ScanNet, el que exista para esa fila).
* HTL/CTL: de los ``'Candidato Valido'`` de Fase 5/5b (``build_traceback_report``
  ya filtra a esos), se colapsan por ``core_9aa`` (mismo nucleo de union
  evaluado en ventanas de posicion vecinas es la misma prediccion, no
  epitopos distintos -- mismo criterio que ``_deduplicate_protein_mode_windows``
  de ``netmhciipan_engine.py``/``netmhcpan_engine.py``, pero aqui colapsando
  TODA la promiscuidad, no solo por trio exacto), quedandose con la mejor
  fila (mas alelos promiscuos, luego menor %Rank; en CTL ademas prioriza
  ``netcleave_c_term_match == True`` primero). Top-N sobre esas filas
  deduplicadas, mismo criterio de orden.

Decision de diseno: el "epitopo" insertado en los bloques HTL/CTL es
``core_9aa`` (el nucleo de union real evaluado por NetMHCIIpan/NetMHCpan),
NO ``sequence_f5`` (la ventana completa de 15 aa o el peptido evaluado
completo) -- practica estandar en literatura de diseno de vacunas
multi-epitopo publicada (encadenar nucleos de union predichos, no las
ventanas completas que los contienen), y mantiene el constructo mas
compacto.

Manejo de solapamientos entre epitopos candidatos: decision FINAL del
usuario de NO fusionar epitopos de CLASES DISTINTAS aunque se solapen en
posicion dentro de la proteina de origen (p. ej. un B-cell que se solapa
con un HTL) -- fusionarlos rompería la semantica de los linkers, ya que
cada bloque espera un peptido de esa clase especifica, no un hibrido. La
fusion INTRA-clase (dos candidatos de la MISMA clase que se solapan) ya la
resuelve la Fase 3 (union anotada de regiones solapadas del mismo tipo de
motor, antes de que las clases se separen en Fase 4b/4c/5/5b) -- no hace
falta resolverla de nuevo aqui.
"""

from typing import List, NamedTuple, Optional, Tuple

import pandas as pd

from src.config.settings import Settings


class _Block(NamedTuple):
    label: str
    rows: List
    intra_linker: str
    sequence_getter: object  # Callable[[row], str]


def _select_bcell_candidates(
    safe_df: pd.DataFrame, algpred_df: pd.DataFrame, stackgly_df: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """Filtra ``safe_df`` por Non-Allergen + sin sequon glicosilado, rankea por mejor score, top-N."""
    if safe_df.empty:
        return safe_df

    non_allergen_seqs = set(algpred_df[algpred_df["algpred_veredicto"] == "Non-Allergen"]["sequence"]) \
        if not algpred_df.empty else set()
    glyco_risky_seqs = set(stackgly_df[stackgly_df["stackglyembed_veredicto"] == "Glicosilado"]["sequence"]) \
        if not stackgly_df.empty else set()

    candidates = safe_df[
        safe_df["sequence"].isin(non_allergen_seqs) & ~safe_df["sequence"].isin(glyco_risky_seqs)
    ].copy()
    if candidates.empty:
        return candidates

    score_cols = [c for c in candidates.columns if c.endswith("_score")]
    candidates["_rank_score"] = candidates[score_cols].max(axis=1, skipna=True) if score_cols else 0.0
    candidates = candidates.sort_values("_rank_score", ascending=False)
    return candidates.head(top_n).drop(columns="_rank_score")


def _dedupe_by_core(candidate_df: pd.DataFrame, sort_columns: List[Tuple[str, bool]]) -> pd.DataFrame:
    """Colapsa filas con el mismo 'core_9aa', quedandose con la 'mejor' segun ``sort_columns``.

    ``sort_columns``: lista de ``(columna, ascending)`` en orden de prioridad
    (primera = criterio principal de desempate).
    """
    if candidate_df.empty:
        return candidate_df
    by_cols = [c for c, _ in sort_columns]
    ascending = [asc for _, asc in sort_columns]
    ordered = candidate_df.sort_values(by=by_cols, ascending=ascending)
    return ordered.drop_duplicates(subset="core_9aa", keep="first")


def _select_htl_candidates(htl_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Colapsa por 'core_9aa' (mejor promiscuidad/%Rank), top-N."""
    if htl_df.empty:
        return htl_df
    sort_columns = [("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(htl_df, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _select_ctl_candidates(ctl_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Colapsa por 'core_9aa', priorizando evidencia de corte NetCleave, luego promiscuidad/%Rank, top-N."""
    if ctl_df.empty:
        return ctl_df
    sort_columns = [("netcleave_c_term_match", False), ("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(ctl_df, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _score_note(row, fields: List[str]) -> str:
    parts = []
    for field in fields:
        value = getattr(row, field, None)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            parts.append(f"{field}={value}")
    return ", ".join(parts)


def assemble_construct(
    safe_df: pd.DataFrame,
    algpred_df: pd.DataFrame,
    stackgly_df: pd.DataFrame,
    htl_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    top_n_per_class: int = None,
    adjuvant_sequence: Optional[str] = None,
) -> Tuple[str, pd.DataFrame]:
    """Selecciona candidatos top-N por clase y ensambla el constructo final.

    Args:
        safe_df: Salida de Fase 4 (``status == 'Segura'``).
        algpred_df: Salida de Fase 4b (``predict_allergenicity``).
        stackgly_df: Salida de Fase 4c (``predict_nglycosylation``).
        htl_df: Salida de Fase 5 (``candidatos_finales.csv`` / ``build_traceback_report``).
        ctl_df: Salida de Fase 5b (``candidatos_finales_mhc1.csv``, con anotacion NetCleave).
        top_n_per_class: Maximo de epitopos por clase (default ``Settings.CONSTRUCT_TOP_N_PER_CLASS``, 3).
        adjuvant_sequence: Secuencia de adjuvante opcional a anteponer en el
            N-terminal (con linker rigido EAAAK). ``None`` por defecto -- ver
            docstring del modulo, ningun adjuvante se elige automaticamente.

    Returns:
        Tupla ``(construct_sequence, metadata_df)``: la secuencia del
        constructo ensamblado (string), y un DataFrame con una fila por
        SEGMENTO (epitopo o linker, en orden) con columnas ``block``,
        ``sequence``, ``start``/``end`` (1-indexado, posicion en el
        constructo final), ``source_accession``, ``source_start``,
        ``source_end`` (``None`` para segmentos de linker/adjuvante) y
        ``source_score_note`` (resumen legible de los scores que motivaron
        la seleccion, vacio para linkers). Concatenar
        ``metadata_df['sequence']`` en orden reconstruye ``construct_sequence``
        exactamente.

        Si las 3 clases quedan vacias tras la seleccion, devuelve
        ``("", DataFrame vacio)`` -- no hay ningun candidato con el cual ensamblar nada.
    """
    top_n = top_n_per_class if top_n_per_class is not None else Settings.CONSTRUCT_TOP_N_PER_CLASS

    bcell_selected = _select_bcell_candidates(safe_df, algpred_df, stackgly_df, top_n)
    htl_selected = _select_htl_candidates(htl_df, top_n)
    ctl_selected = _select_ctl_candidates(ctl_df, top_n)

    blocks: List[_Block] = []
    if not bcell_selected.empty:
        blocks.append(_Block(
            "B-cell", list(bcell_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_BCELL, lambda r: r.sequence,
        ))
    if not htl_selected.empty:
        blocks.append(_Block(
            "HTL", list(htl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_HTL, lambda r: r.core_9aa,
        ))
    if not ctl_selected.empty:
        blocks.append(_Block(
            "CTL", list(ctl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_CTL, lambda r: r.core_9aa,
        ))

    if not blocks and not adjuvant_sequence:
        return "", pd.DataFrame(columns=[
            "block", "sequence", "start", "end", "source_accession",
            "source_start", "source_end", "source_score_note",
        ])

    segments = []
    cursor = 1

    def _add(block_label: str, sequence: str, source_row=None, score_note: str = "") -> None:
        nonlocal cursor
        start = cursor
        end = cursor + len(sequence) - 1
        segments.append(
            {
                "block": block_label,
                "sequence": sequence,
                "start": start,
                "end": end,
                "source_accession": getattr(source_row, "accession", None),
                "source_start": getattr(source_row, "start", None),
                "source_end": getattr(source_row, "end", None),
                "source_score_note": score_note,
            }
        )
        cursor = end + 1

    if adjuvant_sequence:
        _add("Adjuvante", adjuvant_sequence)
        _add("Linker", Settings.CONSTRUCT_LINKER_ADJUVANTE)

    bcell_score_fields = ["bepipred_score", "epidope_score", "discotope_score", "scannet_score"]
    htl_ctl_score_fields = ["n_alelos_promiscuos", "n_alelos_evaluados", "min_rank_el"]

    for block_idx, block in enumerate(blocks):
        score_fields = bcell_score_fields if block.label == "B-cell" else htl_ctl_score_fields
        if block.label == "CTL":
            score_fields = score_fields + ["netcleave_c_term_match", "netcleave_c_term_score"]

        for i, row in enumerate(block.rows):
            _add(block.label, block.sequence_getter(row), row, _score_note(row, score_fields))
            if i < len(block.rows) - 1:
                _add(f"Linker (intra-{block.label})", block.intra_linker)
        if block_idx < len(blocks) - 1:
            _add("Linker (inter-bloque)", Settings.CONSTRUCT_LINKER_INTERBLOQUE)

    construct_sequence = "".join(s["sequence"] for s in segments)
    metadata_df = pd.DataFrame(segments)
    return construct_sequence, metadata_df
