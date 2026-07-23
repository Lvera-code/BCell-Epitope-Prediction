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
  ya filtra a esos), se descartan las ventanas cuyo rango ``start``/``end``
  solapa la posicion ABSOLUTA de un sequon 'Glicosilado' de Fase 4c (ver
  ``_glycosylated_regions``/``_overlaps_glyco_region`` -- mismo mecanismo de
  solapamiento por posicion que usa ``tmbed_engine`` para enmascarar TM/senal
  en Fase 3b). Las sobrevivientes se colapsan por ``core_9aa`` (mismo nucleo
  de union evaluado en ventanas de posicion vecinas es la misma prediccion,
  no epitopos distintos -- mismo criterio que
  ``_deduplicate_protein_mode_windows`` de
  ``netmhciipan_engine.py``/``netmhcpan_engine.py``, pero aqui colapsando
  TODA la promiscuidad, no solo por trio exacto), quedandose con la mejor
  fila (mas alelos promiscuos, luego menor %Rank; en CTL ademas prioriza
  ``netcleave_c_term_match == True`` primero). Top-N sobre esas filas
  deduplicadas, mismo criterio de orden.
* NO se excluye HTL/CTL por alergenicidad (AlgPred2), a diferencia de
  B-cell: AlgPred2 predice potencial IgE/mastocitario, que depende de que el
  epitopo circule INTACTO y expuesto para ser reconocido por un anticuerpo
  -exactamente el escenario B-cell-. Un nucleo de 8-11 aa que vive escondido
  en el surco del MHC nunca circula libre de esa forma, asi que el
  fundamento mecanicista para descartarlo por "alergenico" es mucho mas
  debil -- decision explicita del usuario de no aplicar ese filtro ahi
  (2026-07-23), para no descartar candidatos MHC validos sin una razon
  biologica solida. La exclusion por glicosilacion SI aplica a las 3 clases
  porque su mecanismo (bloqueo fisico por el glicano) no depende de la via
  de reconocimiento.

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
from src.utils.table_format import Column, print_fixed_width_table


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


def _glycosylated_regions(safe_df: pd.DataFrame, stackgly_df: pd.DataFrame) -> pd.DataFrame:
    """Traduce cada sequon 'Glicosilado' de Fase 4c a su posicion ABSOLUTA en la proteina completa.

    ``stackgly_df['sequon_position']`` es 1-indexado pero LOCAL al peptido
    'Segura' padre (ver ``stackglyembed_engine.print_glycosylation_report``),
    no a la proteina completa -- para poder comparar contra el ``start``/``end``
    (absolutos) de ``htl_df``/``ctl_df`` hay que sumarle el ``start`` de ese
    peptido padre en ``safe_df``.

    Returns:
        DataFrame con columnas ``accession``/``start``/``end`` (1-indexado,
        3 residuos del motivo N-X-[S/T]), una fila por sequon glicosilado.
        Vacio si no hay ningun sequon 'Glicosilado'.
    """
    if stackgly_df.empty or safe_df.empty:
        return pd.DataFrame(columns=["accession", "start", "end"])
    glyco = stackgly_df[stackgly_df["stackglyembed_veredicto"] == "Glicosilado"]
    if glyco.empty:
        return pd.DataFrame(columns=["accession", "start", "end"])
    parent = safe_df[["accession", "start", "sequence"]].rename(columns={"start": "parent_start"})
    merged = glyco.merge(parent, on="sequence", how="inner")
    merged["start"] = merged["parent_start"] + merged["sequon_position"] - 1
    merged["end"] = merged["start"] + 2
    return merged[["accession", "start", "end"]]


def _overlaps_glyco_region(row, glyco_regions: pd.DataFrame) -> bool:
    """Indica si ``row`` (con ``accession``/``start``/``end``) solapa algun sequon glicosilado.

    Mismo mecanismo de solapamiento por posicion que
    ``tmbed_engine.discard_overlapping_regions`` usa para enmascarar TM/senal
    en Fase 3b -- acá se aplica a nivel de candidato individual HTL/CTL en vez
    de a la union de Fase 3.
    """
    if glyco_regions.empty:
        return False
    acc_regions = glyco_regions[glyco_regions["accession"] == row["accession"]]
    if acc_regions.empty:
        return False
    return bool(((acc_regions["start"] <= row["end"]) & (acc_regions["end"] >= row["start"])).any())


def _select_htl_candidates(htl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Colapsa por 'core_9aa' (mejor promiscuidad/%Rank), excluye ventanas glicosiladas, top-N.

    Un glicano real sentado sobre un residuo DENTRO del nucleo de 9 aa que se
    mete en el surco del MHC-II puede bloquear fisicamente la union -- a
    diferencia de la alergenicidad (ver docstring de modulo), esta exclusion
    SI aplica a HTL/CTL igual que a B-cell, no es exclusiva de un mecanismo
    de reconocimiento por anticuerpo/IgE.
    """
    if htl_df.empty:
        return htl_df
    candidates = htl_df[~htl_df.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)]
    if candidates.empty:
        return candidates
    sort_columns = [("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _select_ctl_candidates(ctl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Colapsa por 'core_9aa', excluye ventanas glicosiladas, prioriza NetCleave/promiscuidad/%Rank, top-N.

    Mismo criterio de exclusion por glicosilacion que ``_select_htl_candidates``
    (ver ese docstring): un glicano dentro del nucleo de union MHC-I tambien
    puede bloquear el procesamiento/presentacion, sin importar la via.
    """
    if ctl_df.empty:
        return ctl_df
    candidates = ctl_df[~ctl_df.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)]
    if candidates.empty:
        return candidates
    sort_columns = [("netcleave_c_term_match", False), ("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
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

    glyco_regions = _glycosylated_regions(safe_df, stackgly_df)
    bcell_selected = _select_bcell_candidates(safe_df, algpred_df, stackgly_df, top_n)
    htl_selected = _select_htl_candidates(htl_df, glyco_regions, top_n)
    ctl_selected = _select_ctl_candidates(ctl_df, glyco_regions, top_n)

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


def print_construct_breakdown(metadata_df: pd.DataFrame) -> None:
    """Imprime el constructo desglosado: Bloque/Start/End/Origen, un segmento por fila, en orden.

    Analogo al resto de tablas de desglose del pipeline (ver
    ``algpred_engine.print_allergenicity_report``): la consola solo mostraba
    antes la secuencia concatenada del constructo completo, sin ver de que
    peptidos/linkers individuales esta hecho ni de donde salio cada uno. La
    secuencia de cada segmento se ve coloreada en ``print_construct_colored``
    (linkers en rojo), no repetida acá como texto plano -- esta tabla es
    solo la trazabilidad (posicion + origen); el detalle completo (score que
    motivo la seleccion, posicion en la proteina de origen) sigue persistido
    en ``constructo_metadata.csv``.
    """
    if metadata_df.empty:
        return

    def _origen(r) -> str:
        return r.source_accession if pd.notna(r.source_accession) else "-"

    columns = [
        Column("Bloque", lambda r: r.block, 22, "<"),
        Column("Start", lambda r: str(r.start), 6, ">"),
        Column("End", lambda r: str(r.end), 6, ">"),
        # prefix="  ": "End" es right-aligned, sin este separador explicito
        # quedaria pegada a "Origen" (mismo caso que tmbed_engine.py/signalp_engine.py).
        Column("Origen", _origen, 0, "<", prefix="  "),
    ]
    print_fixed_width_table(metadata_df.itertuples(index=False), columns)


# Amarillo/negrita para los linkers dentro del constructo impreso -- MISMO
# color que el nucleo MHC (netmhciipan_engine.py) y el sequon
# (stackglyembed_engine.py), para mantener un unico codigo de resaltado
# consistente en todo el pipeline.
_LINKER_ANSI_START = "\033[1;33m"
_LINKER_ANSI_END = "\033[0m"


def print_construct_colored(metadata_df: pd.DataFrame) -> None:
    """Imprime la secuencia completa del constructo con los linkers en amarillo.

    Complementa ``print_construct_breakdown`` (da start/end/origen por
    segmento pero no la secuencia): acá se ve el constructo entero tal cual
    queda en el FASTA, con cada tramo de linker coloreado para diferenciarlo
    a simple vista de los tramos de epitopo real. No es una tabla de ancho
    fijo -- es una sola linea concatenada, asi que no aplica el problema de
    alineado de ``table_format.Column`` (los codigos ANSI solo importan para
    ``len()`` cuando hay padding de columna de por medio).
    """
    if metadata_df.empty:
        return
    parts = [
        f"{_LINKER_ANSI_START}{row.sequence}{_LINKER_ANSI_END}" if row.block.startswith("Linker") else row.sequence
        for row in metadata_df.itertuples(index=False)
    ]
    print("".join(parts))


def print_multi_accession_warning(metadata_df: pd.DataFrame) -> None:
    """Advierte si el constructo se armo con candidatos de mas de un accession distinto.

    Chequeo simple por CANTIDAD de accessions distintos, sin evaluar similitud
    de secuencia entre ellos -- no distingue "cepas relacionadas del mismo
    patogeno" (ej. distintas cepas de VIH, resultado esperado y correcto) de
    "antigenos sin relacion" (ej. mezclar dos patogenos distintos en un mismo
    FASTA de entrada). Queda a criterio de quien lee la advertencia: revisar
    la columna 'Origen' de ``print_construct_breakdown`` para decidir si el
    resultado tiene sentido biologico.
    """
    accessions = metadata_df["source_accession"].dropna().unique()
    if len(accessions) > 1:
        print(f"\n[AVISO] Constructo realizado a partir de diferentes accessions: {', '.join(sorted(accessions))}")
