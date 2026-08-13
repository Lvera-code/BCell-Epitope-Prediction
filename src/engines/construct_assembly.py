"""Fase 7: ensamblaje automatico del constructo multi-epitopo vacunal.

Toma los candidatos que ya sobrevivieron el resto del pipeline (B-cell de
Fase 4/4b/4c, HTL de Fase 5, CTL de Fase 5b), selecciona los mejores
``Settings.CONSTRUCT_TOP_N_PER_CLASS`` por clase, y los concatena con los
linkers estandar del campo de diseno de vacunas multi-epitopo en un unico
FASTA, junto con una tabla de metadata 100% trazable (que peptido individual
aporto cada tramo, en que orden, que linker se uso en cada union). Logica
CASI 100% pura (sin subprocess): la Fase 6b (``conservation_engine``,
OPCIONAL, solo si el usuario paso ``--panel-conservacion``) corre BLASTp
ANTES de llegar aqui y entrega ya calculado ``conservation_df``, exactamente
igual que ``algpred_df``/``stackgly_df`` -- esta invariante es lo que permite
testear la mayoria de este modulo con DataFrames sinteticos. UNICA excepcion
deliberada: ``_pad_short_bcell_candidates`` SI invoca AlgPred2/StackGlyEmbed
de verdad (re-chequeo de la secuencia extendida con flancos, ver su
docstring) -- justificado porque esa secuencia extendida es contenido nuevo
que Fase 4b/4c nunca evaluo, no algo que ya viniera precalculado.

Reglas de ensamblaje (convencion estandar del campo, con multiples fuentes
que coinciden):

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

Seleccion top-N por clase (necesaria en la practica: una corrida real con
GP120 dio 18 candidatos validos solo en HTL/Fase 5, demasiados para un
constructo manejable):

* B-cell: de ``safe_df`` (Fase 4 'Segura'), TODOS los candidatos entran al
  ranking, incluidos los marcados 'Allergen' por AlgPred2 (Fase 4b).
  DECISION 2026-08-13: hasta entonces se descartaban aqui -- se revirtio
  porque el mismo veredicto de AlgPred2 resulto no corresponder con datos
  clinicos/poblacionales reales en la mayoria de los 9 candidatos
  evaluados de la validacion de publicacion (7 de 9 marcados 'Allergen'
  sin correlato de alergenicidad documentado, correlacionando en cambio
  con el ancho de la region que le llega desde Fase 3, no con la quimica
  real del antigeno) -- excluir automaticamente en base a ese veredicto
  perdia candidatos reales y seguros. Igual que con la glicosilacion
  (parrafo siguiente): existen ademas anticuerpos descritos que
  reconocen regiones con similitud de secuencia a alergenos conocidos
  sin que eso implique riesgo real, asi que ya no hay una razon
  mecanistica universal para excluir. En su lugar, cada candidato se
  anota con su veredicto de alergenicidad (columna ``allergen``, visible
  en ``source_score_note`` del constructo final) para que la decision
  quede informada, no automatica. La N-glicosilacion (StackGlyEmbed,
  Fase 4c) tampoco excluye candidatos, decision mas antigua (2026-08-01):
  existen anticuerpos descritos que reconocen especificamente regiones
  glicosiladas (p. ej. epitopos de envoltura de HIV), asi que descartar
  por glicosilacion perdia candidatos biologicamente validos sin una
  razon mecanistica universal. En su lugar, cada candidato se anota con
  su estado de glicosilacion (columna ``glycosylated``, visible en
  ``source_score_note`` del constructo final) -- un peptido SIN ningun
  sequon en el reporte de Fase 4c se anota como ``glycosylated=False``
  (Fase 4c solo produce filas para sequones reales). El top-N se elige
  por el MAYOR de sus ``'{motor}_score'`` disponibles (BepiPred/EpiDope/
  DiscoTope/ScanNet, el que exista para esa fila) -- sin filtro previo
  por alergenicidad ni glicosilacion. Tambien se anota
  ``documented_region`` (Fase 6c, IEDB): ``True`` si el candidato solapa
  con un epitopo de proteccion/neutralizacion YA DOCUMENTADO en algun
  patogeno estudiado -- puramente informativo, igual que
  ``conservation_pct``, NO influye en el ranking todavia. Solo aplica a
  B-cell (no HTL/CTL, ver docstring de ``src.engines.iedb_engine``).
* HTL/CTL: de los ``'Candidato Valido'`` de Fase 5/5b (``build_traceback_report``
  ya filtra a esos), TODAS las ventanas se mantienen independientemente
  de si solapan un sequon 'Glicosilado' de Fase 4c -- mismo razonamiento
  que B-cell arriba. ``_glycosylated_regions``/``_overlaps_glyco_region``
  (mismo mecanismo de solapamiento por posicion que usa ``tmbed_engine``
  para enmascarar TM/senal en Fase 3b) ahora solo ANOTAN la columna
  ``glycosylated`` por fila, no descartan. Las ventanas se colapsan por
  ``core_9aa`` (mismo nucleo de union evaluado en ventanas de posicion
  vecinas es la misma prediccion, no epitopos distintos -- mismo criterio
  que ``_deduplicate_protein_mode_windows`` de
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
  debil -- deliberadamente no se aplica ese filtro ahi, para no descartar
  candidatos MHC validos sin una razon biologica solida.

Decision de diseno: el "epitopo" insertado en los bloques HTL/CTL es
``sequence_f5`` (la ventana completa evaluada -- 15-mero para HTL, el
peptido evaluado completo para CTL), NO solo ``core_9aa`` (el nucleo de
union real evaluado por NetMHCIIpan/NetMHCpan) -- los residuos
flanqueantes alrededor del nucleo tambien contribuyen al reconocimiento
(estabilidad de la interaccion peptido-MHC/TCR, procesamiento antigenico
correcto), asi que reducir el bloque insertado al nucleo minimo
descartaba esa contribucion sin necesidad. ``core_9aa`` se sigue usando
para deduplicar candidatos solapados (mismo nucleo de union = misma
prediccion, sea cual sea la ventana completa que lo contiene) y queda
registrado en la trazabilidad, pero la secuencia que entra al constructo
final es la ventana completa.

Manejo de solapamientos entre epitopos candidatos: decision FINAL del
usuario de NO fusionar epitopos de CLASES DISTINTAS aunque se solapen en
posicion dentro de la proteina de origen (p. ej. un B-cell que se solapa
con un HTL) -- fusionarlos rompería la semantica de los linkers, ya que
cada bloque espera un peptido de esa clase especifica, no un hibrido. La
fusion INTRA-clase (dos candidatos de la MISMA clase que se solapan) ya la
resuelve la Fase 3 (union anotada de regiones solapadas del mismo tipo de
motor, antes de que las clases se separen en Fase 4b/4c/5/5b) -- no hace
falta resolverla de nuevo aqui.

Largo maximo de candidatos B-cell (``Settings.CONSTRUCT_BCELL_MAX_LENGTH``,
20 aa): a diferencia de HTL/CTL (ventana de tamano fijo impuesta por
NetMHCIIpan/NetMHCpan), un candidato B-cell viene de la union anotada de
Fase 3, que puede fusionar transitivamente regiones solapadas de varios
motores en una sola secuencia de decenas de aa -- mas larga que un epitopo
lineal B-cell tipico, y sin evaluar individualmente cada sub-tramo por
AlgPred2/glicosilacion. Si el candidato seleccionado supera el maximo,
``_trim_long_bcell_candidates`` lo recorta a la sub-ventana de mayor score
por-residuo (releyendo los raw CSV de Fase 2, mismo criterio anti-sesgo-de-
escala de percentil-por-motor que el ranking de ``_select_bcell_candidates``)
en vez de insertar la region fusionada completa. El recorte ocurre DESPUES
del top-N (el ranking sigue siendo por la fuerza de la region completa), y
queda registrado en ``trimmed_from_length`` para trazabilidad.

Flancos nativos en candidatos B-cell CORTOS (``Settings.
CONSTRUCT_BCELL_FLANK_THRESHOLD``, 15 aa / ``Settings.
CONSTRUCT_BCELL_FLANK_PADDING``, 3 aa por lado): un candidato B-cell en el
piso de largo (9 aa, ``MIN_FINAL_PEPTIDE_LENGTH`` de ``consensus.py``) NO
tiene el mismo colchon de contexto que HTL/CTL -- ahi, los flancos de
``sequence_f5`` YA fueron evaluados por NetMHCIIpan/NetMHCpan junto al
nucleo (evidencia computacional real, solo decidimos no descartarla). Los
motores B-cell no tienen esa estructura de dos niveles: el limite de la
union de Fase 3 YA ES el limite completo de lo que cualquier motor marco
como antigenico -- no hay un "core" mas chico escondido dentro de una
ventana mas grande que el motor haya evaluado y no usemos. Extender un
candidato corto agrega residuos nativos que NINGUN motor flageo como
antigenicos: es una asuncion de DISEÑO (contexto estructural/exposicion del
paratopo mas alla de un umbral estadistico arbitrario, practica reconocida
en literatura de vacunas multiepitopo), no un hallazgo de herramienta como
el resto de este pipeline -- por eso queda anotada aparte
(``flanked_from_length``) y NO se mezcla con el mismo nivel de certeza que
``bepipred_score``/etc. ``_pad_short_bcell_candidates`` re-chequea la
secuencia YA extendida con AlgPred2/StackGlyEmbed de verdad (es contenido
nuevo, Fase 4b/4c nunca lo vio) y anota el veredicto de alergenicidad de
la version extendida en la columna ``allergen`` -- el padding SIEMPRE se
aplica, ya no se descarta si la version extendida sale 'Allergen'
(DECISION 2026-08-13, revierte la invariante previa: alergenicidad ya no
excluye nada en B-cell, ver parrafo de mas arriba). La glicosilacion,
igual que antes, solo se re-anota (nunca excluyo en B-cell, ver decision
de 2026-08-01). Aplica solo a candidatos <15 aa: los que ya miden 15-20 aa
no se tocan (no hay evidencia de que les falte contexto), y los que
superan 20 aa van al recorte de arriba, nunca a padding (rangos disjuntos
por construccion).
"""

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from src.config.settings import Settings
from src.engines import bepipred_engine, discotope_engine, epidope_engine, scannet_engine
from src.engines.algpred_engine import predict_allergenicity
from src.engines.blast_engine import filter_self_tolerant
from src.engines.stackglyembed_engine import predict_nglycosylation
from src.utils.table_format import Column, print_fixed_width_table

# Motores que producen score POR-RESIDUO cacheado en '{input_stem}_{motor}_raw.csv'
# (mismo nombre de archivo que escribe '_cached_raw_scores'/'_cached_structural_raw_scores'
# en pipeline.py) -- usado por '_trim_long_bcell_candidates' para encontrar la
# sub-ventana de mayor score real dentro de una region B-cell fusionada
# demasiado larga. Reusa las constantes ACCESSION_COLUMN/SCORE_COLUMN de cada
# motor en vez de hardcodear nombres de columna, para no desincronizarse si un
# motor renombra su columna de salida.
_RAW_SCORE_ENGINES = {
    "bepipred": (bepipred_engine.ACCESSION_COLUMN, bepipred_engine.SCORE_COLUMN),
    "epidope": (epidope_engine.ACCESSION_COLUMN, epidope_engine.SCORE_COLUMN),
    "discotope": (discotope_engine.ACCESSION_COLUMN, discotope_engine.SCORE_COLUMN),
    "scannet": (scannet_engine.ACCESSION_COLUMN, scannet_engine.SCORE_COLUMN),
}


class _Block(NamedTuple):
    label: str
    rows: List
    intra_linker: str
    sequence_getter: object  # Callable[[row], str]


def _load_per_residue_scores(output_dir: Path, input_stem: str, accession: str) -> pd.DataFrame:
    """Combina el score por-residuo de ``accession`` entre todos los motores con raw CSV cacheado.

    Cada motor contribuye su score normalizado a percentil DENTRO de su
    propia columna (mismo criterio anti-sesgo-de-escala que el ranking de
    ``_select_bcell_candidates``), promediado entre motores disponibles.
    Los CSV crudos ya estan cacheados por Fase 2 (``_cached_raw_scores``/
    ``_cached_structural_raw_scores`` en ``pipeline.py``): esta funcion solo
    los relee, no vuelve a correr ningun motor. La posicion (1-indexada) se
    infiere del ORDEN de las filas dentro de cada ``accession`` -- los CSV
    crudos de estos 4 motores no traen una columna de posicion explicita,
    solo residuo en orden de secuencia.

    Returns:
        DataFrame con columnas ``position``/``combined_score``, o vacio si
        ningun motor tiene raw CSV cacheado para este ``input_stem`` (p. ej.
        si Fase 2 aun no corrio, lo cual no deberia pasar en Fase 7).
    """
    per_engine_percentiles = []
    for engine_key, (accession_col, score_col) in _RAW_SCORE_ENGINES.items():
        raw_path = output_dir / f"{input_stem}_{engine_key}_raw.csv"
        if not raw_path.is_file():
            continue
        raw_df = pd.read_csv(raw_path)
        if accession_col not in raw_df.columns or score_col not in raw_df.columns:
            continue
        acc_rows = raw_df[raw_df[accession_col] == accession].reset_index(drop=True)
        if acc_rows.empty:
            continue
        percentile = acc_rows[score_col].rank(pct=True)
        percentile.index = acc_rows.index + 1  # posicion 1-indexada
        per_engine_percentiles.append(percentile)

    if not per_engine_percentiles:
        return pd.DataFrame(columns=["position", "combined_score"])

    combined = pd.concat(per_engine_percentiles, axis=1).mean(axis=1, skipna=True)
    return combined.rename("combined_score").rename_axis("position").reset_index()


def _best_subwindow(sequence: str, start: int, per_residue: pd.DataFrame, max_length: int) -> Tuple[str, int, int]:
    """Recorta ``sequence`` (que arranca en la posicion absoluta ``start``) a ``max_length`` aa.

    Desliza una ventana de ``max_length`` y elige la de mayor score
    combinado (suma de ``per_residue['combined_score']`` dentro del rango).
    Si ``per_residue`` esta vacio (ningun motor con raw cacheado para esa
    ``accession`` -- no deberia ocurrir en uso normal), recorta centrado
    como ultimo recurso en vez de fallar.

    Returns:
        Tupla ``(sub_secuencia, nuevo_start, nuevo_end)``, ambos 1-indexados
        y absolutos (misma convencion que ``start``/``end`` de ``safe_df``).
    """
    length = len(sequence)
    if length <= max_length:
        return sequence, start, start + length - 1

    if per_residue.empty:
        offset = (length - max_length) // 2
    else:
        scores = per_residue.set_index("position")["combined_score"]
        window_sums = [
            scores.reindex(range(start + i, start + i + max_length)).sum(skipna=True)
            for i in range(length - max_length + 1)
        ]
        offset = max(range(len(window_sums)), key=lambda i: window_sums[i])

    new_start = start + offset
    new_end = new_start + max_length - 1
    return sequence[offset:offset + max_length], new_start, new_end


def _load_native_residues(output_dir: Path, input_stem: str, accession: str) -> Optional[pd.Series]:
    """Devuelve la secuencia nativa completa de ``accession`` (posicion 1-indexada -> residuo).

    Relee el primer raw CSV de Fase 2 disponible para esa ``accession`` --
    misma fuente que ``_load_per_residue_scores``, no una copia independiente
    (evita que padding y scoring se desincronicen si algun motor difiere en
    que posiciones parseo, ver ADR de ``position_mapping`` en
    ``consensus.py``). Sirve tanto para el camino FASTA (BepiPred/EpiDope)
    como estructura (DiscoTope/ScanNet, cadena unica) -- los 4 raw CSV traen
    una columna ``Residue`` con el aminoacido en cada posicion.

    Returns:
        ``pd.Series`` indexada por posicion 1-indexada, o ``None`` si ningun
        motor tiene raw CSV cacheado para esa accession (no deberia ocurrir
        en uso normal).
    """
    for engine_key, (accession_col, _) in _RAW_SCORE_ENGINES.items():
        raw_path = output_dir / f"{input_stem}_{engine_key}_raw.csv"
        if not raw_path.is_file():
            continue
        raw_df = pd.read_csv(raw_path)
        if accession_col not in raw_df.columns or "Residue" not in raw_df.columns:
            continue
        acc_rows = raw_df[raw_df[accession_col] == accession].reset_index(drop=True)
        if acc_rows.empty:
            continue
        residues = acc_rows["Residue"].copy()
        residues.index = acc_rows.index + 1  # posicion 1-indexada
        return residues
    return None


def _pad_short_bcell_candidates(
    candidates: pd.DataFrame, output_dir: Path, input_stem: str, threshold: int, padding: int,
) -> pd.DataFrame:
    """Extiende con residuos nativos flanqueantes cada candidato B-cell mas corto que ``threshold``.

    Ver la seccion "Flancos nativos..." del docstring del modulo para el
    razonamiento completo (por que esto NO es lo mismo que ``sequence_f5``
    en HTL/CTL). Resumen operativo:

    1. Para cada candidato con ``len(sequence) < threshold``, propone
       extenderlo ``padding`` aa a cada lado usando la secuencia nativa real
       (``_load_native_residues``), recortado a los limites de la proteina.
    2. Re-chequea TODAS las propuestas de una sola vez con AlgPred2
       (``predict_allergenicity``) y StackGlyEmbed (``predict_nglycosylation``)
       -- unica excepcion de este modulo a "sin subprocess real", ver
       docstring del modulo.
    3. El padding SIEMPRE se aplica (DECISION 2026-08-13): el veredicto de
       alergenicidad de la version extendida se anota en ``allergen``, igual
       que la glicosilacion se anota en ``glycosylated`` -- ninguna de las
       dos excluye nada en B-cell.

    Candidatos ya en el borde de la proteina (``new_start == start`` y
    ``new_end == end``, no hay hacia donde extender) se dejan sin tocar. Si
    ``_load_native_residues`` no encuentra raw cacheado para una accession,
    ese candidato tambien se deja sin tocar (mismo fallback conservador que
    ``_trim_long_bcell_candidates``: preferir no tocar antes que fallar).

    Anota ``flanked_from_length`` (largo original antes del padding, ``NaN``
    si no se extendio) para trazabilidad en ``source_score_note``.
    """
    if candidates.empty or output_dir is None or input_stem is None:
        return candidates

    padded = candidates.copy()
    short_rows = padded[padded["sequence"].str.len() < threshold]
    if short_rows.empty:
        padded["flanked_from_length"] = pd.NA
        return padded

    proposals: Dict[int, Tuple[str, int, int]] = {}
    for idx, row in short_rows.iterrows():
        residues = _load_native_residues(output_dir, input_stem, row["accession"])
        if residues is None or residues.empty:
            continue
        new_start = max(int(residues.index.min()), int(row["start"]) - padding)
        new_end = min(int(residues.index.max()), int(row["end"]) + padding)
        if new_start == row["start"] and new_end == row["end"]:
            continue
        new_seq = "".join(residues.loc[new_start:new_end])
        proposals[idx] = (new_seq, new_start, new_end)

    padded["flanked_from_length"] = pd.NA
    if not proposals:
        return padded

    proposal_seqs = sorted({seq for seq, _, _ in proposals.values()})
    algpred_check = predict_allergenicity(proposal_seqs, output_dir, filename_prefix=f"{input_stem}_bcell_flank_")
    glyco_check = predict_nglycosylation(proposal_seqs, output_dir, filename_prefix=f"{input_stem}_bcell_flank_")
    allergen_seqs = set(algpred_check[algpred_check["algpred_veredicto"] == "Allergen"]["sequence"]) \
        if not algpred_check.empty else set()
    glyco_risky_seqs = set(glyco_check[glyco_check["stackglyembed_veredicto"] == "Glicosilado"]["sequence"]) \
        if not glyco_check.empty else set()

    for idx, (new_seq, new_start, new_end) in proposals.items():
        original_length = len(padded.at[idx, "sequence"])
        padded.at[idx, "sequence"] = new_seq
        padded.at[idx, "start"] = new_start
        padded.at[idx, "end"] = new_end
        padded.at[idx, "allergen"] = new_seq in allergen_seqs
        padded.at[idx, "glycosylated"] = new_seq in glyco_risky_seqs
        padded.at[idx, "flanked_from_length"] = original_length

    return padded


def _trim_long_bcell_candidates(
    candidates: pd.DataFrame, output_dir: Path, input_stem: str, max_length: int,
) -> pd.DataFrame:
    """Recorta cada candidato B-cell que supere ``max_length`` a su mejor sub-ventana (ver ``_best_subwindow``).

    Solo se aplica a los candidatos YA SELECCIONADOS (top-N), no a todo
    ``safe_df``: el ranking sigue siendo por la region fusionada completa
    (``_select_bcell_candidates``), el recorte es puramente para la
    secuencia que efectivamente entra al constructo. Anota
    ``trimmed_from_length`` (largo original, ``NaN`` si no se recorto) para
    trazabilidad en ``source_score_note``.
    """
    if candidates.empty:
        return candidates
    trimmed = candidates.copy()
    original_lengths = trimmed["sequence"].str.len()
    for idx, row in trimmed.iterrows():
        if len(row["sequence"]) <= max_length:
            continue
        per_residue = _load_per_residue_scores(output_dir, input_stem, row["accession"])
        new_seq, new_start, new_end = _best_subwindow(row["sequence"], int(row["start"]), per_residue, max_length)
        trimmed.at[idx, "sequence"] = new_seq
        trimmed.at[idx, "start"] = new_start
        trimmed.at[idx, "end"] = new_end
    trimmed["trimmed_from_length"] = original_lengths.where(original_lengths > max_length)
    return trimmed


def _select_bcell_candidates(
    safe_df: pd.DataFrame, algpred_df: pd.DataFrame, stackgly_df: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
    iedb_matched_seqs: Optional[set] = None,
    output_dir: Optional[Path] = None,
    input_stem: Optional[str] = None,
    max_length: Optional[int] = None,
    flank_threshold: Optional[int] = None,
    flank_padding: Optional[int] = None,
    blast_db: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
) -> pd.DataFrame:
    """Anota ``safe_df`` con alergenicidad/glicosilacion/conservacion/region-documentada (sin excluir), rankea por consenso entre motores, top-N.

    DECISION 2026-08-13: ya no filtra por 'Non-Allergen' -- todos los
    candidatos de ``safe_df`` entran al ranking, anotados con su veredicto
    de AlgPred2 en la columna ``allergen`` (``True`` = 'Allergen'). Ver el
    parrafo de alergenicidad en el docstring del modulo para el porque.

    El ranking usa el percentil de cada candidato DENTRO de la columna
    ``{motor}_score`` a la que pertenece (``rank(pct=True)``), no el score
    crudo: los motores no comparten escala (DiscoTope-3.0 es un score
    calibrado que puede superar 1, mientras BepiPred/EpiDope/ScanNet son
    probabilidades en [0,1]), asi que promediar/maximizar los valores crudos
    sesgaba el ranking hacia el motor con el rango numerico mas grande, no
    hacia el candidato mas fuerte. El promedio de percentiles (``mean``, no
    ``max``) premia el CONSENSO entre motores: un candidato visto como fuerte
    por varios motores le gana a uno que un solo motor ve como excepcional.
    """
    if safe_df.empty:
        return safe_df

    allergen_seqs = set(algpred_df[algpred_df["algpred_veredicto"] == "Allergen"]["sequence"]) \
        if not algpred_df.empty else set()
    glyco_risky_seqs = set(stackgly_df[stackgly_df["stackglyembed_veredicto"] == "Glicosilado"]["sequence"]) \
        if not stackgly_df.empty else set()

    candidates = safe_df.copy()
    if candidates.empty:
        return candidates
    candidates["allergen"] = candidates["sequence"].isin(allergen_seqs)
    candidates["glycosylated"] = candidates["sequence"].isin(glyco_risky_seqs)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence"].map(conservation_map)
    if iedb_matched_seqs:
        candidates["documented_region"] = candidates["sequence"].isin(iedb_matched_seqs)

    score_cols = [c for c in candidates.columns if c.endswith("_score")]
    if score_cols:
        percentiles = candidates[score_cols].rank(pct=True, na_option="keep")
        candidates["_rank_score"] = percentiles.mean(axis=1, skipna=True)
    else:
        candidates["_rank_score"] = 0.0
    candidates = candidates.sort_values("_rank_score", ascending=False)
    selected = candidates.head(top_n).drop(columns="_rank_score")

    if output_dir is not None and input_stem is not None:
        selected = _pad_short_bcell_candidates(
            selected, output_dir, input_stem,
            flank_threshold or Settings.CONSTRUCT_BCELL_FLANK_THRESHOLD,
            flank_padding or Settings.CONSTRUCT_BCELL_FLANK_PADDING,
        )
        selected = _trim_long_bcell_candidates(
            selected, output_dir, input_stem, max_length or Settings.CONSTRUCT_BCELL_MAX_LENGTH
        )
        # Re-chequeo de autotolerancia sobre la secuencia FINAL (ya recortada/extendida),
        # DESPUES de pad/trim: ni la Fase 4 original (corrio sobre la region padre, potencialmente
        # mas larga) ni el padding de flancos (solo re-chequea AlgPred2/StackGlyEmbed) cubren esto
        # -- ver docstring de 'blast_engine.filter_self_tolerant'.
        selected = filter_self_tolerant(selected, "sequence", db_path=blast_db, identity_threshold=identity_threshold)
    return selected


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


def _select_htl_candidates(
    htl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Colapsa por 'core_9aa' (mejor promiscuidad/%Rank), anota glicosilacion/conservacion (sin excluir), top-N.

    Un glicano dentro del nucleo de 9 aa que se mete en el surco del MHC-II
    podria en teoria interferir con la union, pero existen anticuerpos/celulas
    T descritos que reconocen especificamente epitopos glicosilados (mismo
    razonamiento que B-cell, ver docstring de modulo) -- ya no se descarta por
    esto, solo se anota en la columna ``glycosylated`` para decision informada.
    ``conservation_map`` (``sequence_f5`` -> ``conservation_pct``, de Fase 6b)
    se mapea igual, ausente si no se paso ``--panel-conservacion``.
    """
    if htl_df.empty:
        return htl_df
    candidates = htl_df.copy()
    candidates["glycosylated"] = candidates.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence_f5"].map(conservation_map)
    sort_columns = [("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _select_ctl_candidates(
    ctl_df: pd.DataFrame, glyco_regions: pd.DataFrame, top_n: int,
    conservation_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Colapsa por 'core_9aa', anota glicosilacion/conservacion (sin excluir), prioriza NetCleave/promiscuidad/%Rank, top-N.

    Mismo criterio que ``_select_htl_candidates`` (ver ese docstring): ya no
    se descarta por glicosilacion, solo se anota; mismo mapeo opcional de
    conservacion.
    """
    if ctl_df.empty:
        return ctl_df
    candidates = ctl_df.copy()
    candidates["glycosylated"] = candidates.apply(lambda r: _overlaps_glyco_region(r, glyco_regions), axis=1)
    if conservation_map:
        candidates["conservation_pct"] = candidates["sequence_f5"].map(conservation_map)
    sort_columns = [("netcleave_c_term_match", False), ("n_alelos_promiscuos", False), ("min_rank_el", True)]
    deduped = _dedupe_by_core(candidates, sort_columns)
    deduped = deduped.sort_values(by=[c for c, _ in sort_columns], ascending=[a for _, a in sort_columns])
    return deduped.head(top_n)


def _score_note(row, fields: List[str]) -> str:
    parts = []
    for field in fields:
        value = getattr(row, field, None)
        # pd.isna() cubre tanto NaN de float (ej. 'trimmed_from_length', via
        # Series.where) como pd.NA (ej. 'flanked_from_length') -- el chequeo
        # anterior (isinstance(value, float)) solo filtraba el primero.
        if value is not None and not pd.isna(value):
            parts.append(f"{field}={value}")
    return ", ".join(parts)


def assemble_construct(
    safe_df: pd.DataFrame,
    algpred_df: pd.DataFrame,
    stackgly_df: pd.DataFrame,
    htl_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    input_stem: Optional[str] = None,
    top_n_per_class: int = None,
    bcell_max_length: Optional[int] = None,
    bcell_flank_threshold: Optional[int] = None,
    bcell_flank_padding: Optional[int] = None,
    adjuvant_sequence: Optional[str] = None,
    conservation_df: Optional[pd.DataFrame] = None,
    iedb_df: Optional[pd.DataFrame] = None,
    blast_db: str = Settings.BLAST_HUMAN_DB,
    identity_threshold: float = Settings.BLAST_IDENTITY_THRESHOLD,
) -> Tuple[str, pd.DataFrame]:
    """Selecciona candidatos top-N por clase y ensambla el constructo final.

    Args:
        safe_df: Salida de Fase 4 (``status == 'Segura'``).
        algpred_df: Salida de Fase 4b (``predict_allergenicity``).
        stackgly_df: Salida de Fase 4c (``predict_nglycosylation``).
        htl_df: Salida de Fase 5 (``candidatos_finales.csv`` / ``build_traceback_report``).
        ctl_df: Salida de Fase 5b (``candidatos_finales_mhc1.csv``, con anotacion NetCleave).
        output_dir: Carpeta de ``outputs`` -- SOLO se usa para releer los
            raw CSV por-residuo de Fase 2 y recortar candidatos B-cell que
            excedan ``Settings.CONSTRUCT_BCELL_MAX_LENGTH`` (ver
            ``_trim_long_bcell_candidates``). ``None`` desactiva el recorte
            (compatibilidad con llamadas/tests que no lo necesitan).
        input_stem: Nombre del archivo de entrada sin extension, para ubicar
            esos mismos raw CSV (``{input_stem}_{motor}_raw.csv``).
        top_n_per_class: Maximo de epitopos por clase (default ``Settings.CONSTRUCT_TOP_N_PER_CLASS``, 3).
        bcell_max_length: Largo maximo de un candidato B-cell antes de
            recortarse a su mejor sub-ventana (default
            ``Settings.CONSTRUCT_BCELL_MAX_LENGTH``, 20). Sin efecto si
            ``output_dir``/``input_stem`` son ``None`` (recorte desactivado).
        bcell_flank_threshold: Largo por debajo del cual un candidato B-cell
            se extiende con flancos nativos (default ``Settings.
            CONSTRUCT_BCELL_FLANK_THRESHOLD``, 15). bcell_flank_padding:
            cuantos aa nativos agregar a cada lado (default ``Settings.
            CONSTRUCT_BCELL_FLANK_PADDING``, 3). Ver
            ``_pad_short_bcell_candidates`` -- re-chequea la version
            extendida con AlgPred2/StackGlyEmbed de verdad, unica excepcion
            de este modulo a "sin subprocess". Sin efecto si
            ``output_dir``/``input_stem`` son ``None``.
        adjuvant_sequence: Secuencia de adjuvante opcional a anteponer en el
            N-terminal (con linker rigido EAAAK). ``None`` por defecto -- ver
            docstring del modulo, ningun adjuvante se elige automaticamente.
        conservation_df: Salida OPCIONAL de Fase 6b (``conservation_engine``),
            con columnas ``sequence``/``conservation_pct``. ``None`` por
            defecto (el usuario no paso ``--panel-conservacion``) -- en ese
            caso no se anota nada, mismo comportamiento que antes de este
            parametro. Igual que ``glycosylated``, es puramente informativo
            (visible en ``source_score_note``), NO influye en la seleccion
            top-N: la decision de usarlo como criterio de ranking queda
            pendiente de una sesion futura (ver vault).
        iedb_df: Salida de Fase 6c (``iedb_engine``, siempre corre), con
            columnas ``sequence``/``epitope_sequence``/... Anota
            ``documented_region`` (bool) SOLO en candidatos B-cell -- IEDB
            aca es especificamente ensayos de anticuerpo, no aplica al
            mecanismo de HTL/CTL (ver docstring de ``iedb_engine``). Igual
            que ``conservation_pct``, puramente informativo.
        blast_db/identity_threshold: Mismos parametros que Fase 4, para el
            re-chequeo de autotolerancia del candidato B-cell YA recortado/
            extendido (``filter_self_tolerant``, aplicado DESPUES de
            ``bcell_max_length``/``bcell_flank_*`` dentro de
            ``_select_bcell_candidates``). A diferencia de esos otros
            parametros, este SI puede excluir un candidato del top-N (no es
            informativo): Fase 4 corrio sobre la region padre, potencialmente
            mucho mas larga, asi que no cubre la secuencia final por si sola.
            Sin efecto si ``output_dir``/``input_stem`` son ``None``.

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
    conservation_map = (
        dict(zip(conservation_df["sequence"], conservation_df["conservation_pct"]))
        if conservation_df is not None and not conservation_df.empty else None
    )
    iedb_matched_seqs = (
        set(iedb_df["sequence"]) if iedb_df is not None and not iedb_df.empty else None
    )

    glyco_regions = _glycosylated_regions(safe_df, stackgly_df)
    bcell_selected = _select_bcell_candidates(
        safe_df, algpred_df, stackgly_df, top_n, conservation_map, iedb_matched_seqs,
        output_dir=output_dir, input_stem=input_stem, max_length=bcell_max_length,
        flank_threshold=bcell_flank_threshold, flank_padding=bcell_flank_padding,
        blast_db=blast_db, identity_threshold=identity_threshold,
    )
    htl_selected = _select_htl_candidates(htl_df, glyco_regions, top_n, conservation_map)
    ctl_selected = _select_ctl_candidates(ctl_df, glyco_regions, top_n, conservation_map)

    blocks: List[_Block] = []
    if not bcell_selected.empty:
        blocks.append(_Block(
            "B-cell", list(bcell_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_BCELL, lambda r: r.sequence,
        ))
    if not htl_selected.empty:
        blocks.append(_Block(
            "HTL", list(htl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_HTL, lambda r: r.sequence_f5,
        ))
    if not ctl_selected.empty:
        blocks.append(_Block(
            "CTL", list(ctl_selected.itertuples(index=False)),
            Settings.CONSTRUCT_LINKER_CTL, lambda r: r.sequence_f5,
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

    bcell_score_fields = [
        "bepipred_score", "epidope_score", "discotope_score", "scannet_score",
        "allergen", "glycosylated", "conservation_pct", "documented_region",
        "trimmed_from_length", "flanked_from_length",
    ]
    htl_ctl_score_fields = [
        "n_alelos_promiscuos", "n_alelos_evaluados", "population_coverage_pct",
        "min_rank_el", "glycosylated", "conservation_pct",
    ]

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
