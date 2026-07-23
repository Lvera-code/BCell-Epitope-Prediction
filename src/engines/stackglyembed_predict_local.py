"""Extraccion de features + prediccion de N-glicosilacion (StackGlyEmbed), 100% local/offline.

Adaptacion de ``extractFeatures.py`` + ``predict.py`` del repo original
(github.com/GaryChan-lab/StackGlyEmbed, clonado en ``StackGlyEmbed/`` --
ignorado por git, ver ``.gitignore``) para uso desde
``src/engines/stackglyembed_engine.py``. Este script vive en el arbol
versionado del proyecto (NO dentro del clon ignorado) porque es codigo
propio, no parte del repo original: ``StackGlyEmbed/`` tiene su propio
``.git`` (es un repo anidado), y git no permite "des-ignorar" un archivo
dentro de un repo anidado con ningun patron de ``.gitignore``. Los
pickles del clasificador (``power_transformer_*.sav``,
``base_layer_pickle_files/``) SI viven dentro del clon ignorado y se
referencian por ``--models-dir`` (nunca hardcodeados).

Cambios respecto al original:

1. **ESM-2 via ``transformers.EsmModel`` en vez de ``torch.hub.load``**: el
   original llama a ``torch.hub.load("facebookresearch/esm:main", ...)``, que
   golpea red EN CADA CORRIDA (el hub de torch no cachea el codigo de forma
   reutilizable sin red de la misma manera que HF Hub). ``facebook/esm2_t33_650M_UR50D``
   via ``transformers`` es el mismo modelo (mismo autor, mismo checkpoint
   original), soporta carga 100% offline una vez cacheado
   (``HF_HUB_OFFLINE=1``, ver mas abajo), y produce las mismas
   representaciones por capa (33 capas, dim 1280) que el codigo original
   consume.

2. **ProtT5 apunta a una ruta LOCAL** (por defecto, la carpeta de pesos ya
   descargada para TMbed en ``scipion-chem-tmbed/tmbed_src/tmbed/models/t5/``,
   modelo ``Rostlab/prot_t5_xl_half_uniref50-enc`` -- mismo encoder que
   ``Rostlab/prot_t5_xl_uniref50`` que pedia el script original, solo que en
   fp16/sin decoder-, en vez de descargarlo de nuevo) en vez del ID remoto
   ``Rostlab/prot_t5_xl_uniref50`` que el original resolvia siempre contra el
   Hub.

3. **ProteinBERT sin cambios de fondo**: ``load_pretrained_model()`` ya es
   offline-segura una vez que ``~/proteinbert_models/default.pkl`` existe (se
   descarga una unica vez como paso de instalacion) -- solo se le
   pasa ``download_model_dump_if_not_exists=False`` aqui para que *falle* con
   un error claro en vez de intentar red si por alguna razon el dump no
   esta, en lugar de descargar silenciosamente en medio de una corrida del
   pipeline.

4. **Rutas de entrada/salida/modelos por CLI** en vez de nombres de archivo
   hardcodeados relativos al cwd (``dataset.txt``, ``features.csv``,
   ``predicted_values.txt``, pickles en ``./base_layer_pickle_files/`` en el
   original): permite correr multiples invocaciones concurrentes/sucesivas
   sin pisarse y sin depender de que el cwd sea una carpeta especifica del
   clon externo.

5. **La embedding global de ProteinBERT se calcula UNA VEZ por proteina**
   (fuera del loop de sitios), no una vez por cada sitio candidato como en el
   original (``getRepresentation`` se llamaba dentro del loop de ``site_position``,
   recalculando el mismo forward pass determinista N veces para N sitios de
   la misma proteina). Resultado numerico identico, solo evita trabajo
   redundante.

El formato de ``dataset.txt`` (``Protein_id,site_1,site_2,...`` seguido de la
secuencia en la linea siguiente) y el orden/dimensionalidad de columnas de
``features.csv`` (ProteinBERT global + ESM-2 promediado en ventana +
ProtT5 residuo puntual, en ese orden) se mantienen EXACTOS respecto al
original: los clasificadores de ``base_layer_pickle_files/`` y los
``power_transformer_*.sav`` fueron entrenados contra ese orden preciso y no
se tocan aqui.
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pickle  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import PowerTransformer  # noqa: E402,F401 (necesario para des-picklear los power_transformer_*.sav)
from tensorflow import keras  # noqa: E402
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer  # noqa: E402

from proteinbert import load_pretrained_model  # noqa: E402

_WINDOW_SIZE = 15
_DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
_DEFAULT_T5_MODEL_PATH = str(
    Path.home() / "DiffSBDD" / "scipion-chem-tmbed" / "tmbed_src" / "tmbed" / "models" / "t5"
)


def _get_model_with_global_embedding_as_outputs(model):
    """Reconstruye el modelo de ProteinBERT para exponer la embedding global (ver README/extractFeatures.py original)."""
    global_layers = [
        layer.output
        for layer in model.layers
        if len(layer.output.shape) == 2 and layer.name in ["global-merge2-norm-block6"]
    ]
    concatenated = keras.layers.Concatenate(name="last-Window-layers")(global_layers)
    return keras.models.Model(inputs=model.inputs, outputs=concatenated)


def _get_proteinbert_representation(pretrained_model_generator, input_encoder, seq: str) -> np.ndarray:
    encoded_x = input_encoder.encode_X([seq], len(seq) + 2)
    model = _get_model_with_global_embedding_as_outputs(pretrained_model_generator.create_model(len(seq) + 2))
    return np.array(model.predict(encoded_x, batch_size=2))[0]


def _get_esm2_embedding(tokenizer, model, seq: str) -> np.ndarray:
    """Embedding ESM-2 por residuo (representaciones de la ultima capa, sin CLS/EOS)."""
    chunks = [seq[i : i + 1024] for i in range(0, len(seq), 1024)]
    final = np.zeros((1, model.config.hidden_size))
    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            out = model(**tokens)
        rep = out.last_hidden_state[0, 1:-1].numpy()
        final = np.concatenate((final, rep), axis=0)
    return np.delete(final, 0, axis=0)


def _get_prott5_embedding(tokenizer, model, seq: str) -> np.ndarray:
    """Embedding ProtT5 por residuo (mismo chunking de 8797 aa que el script original)."""
    chunks = [seq[i : i + 8797] for i in range(0, len(seq), 8797)]
    final = np.zeros((1, model.config.d_model))
    for chunk in chunks:
        spaced = " ".join(list(re.sub(r"[UZOB]", "X", chunk)))
        ids = tokenizer([spaced], add_special_tokens=True, padding="longest", return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
        emb = out.last_hidden_state[0, : len(chunk)].numpy()
        final = np.concatenate((final, emb), axis=0)
    return np.delete(final, 0, axis=0)


def extract_features(dataset_path: Path, output_dir: Path, t5_model_path: str, esm_model_name: str) -> Path:
    """Genera ``features.csv`` (ProteinBERT + ESM-2 + ProtT5) para cada sitio de ``dataset.txt``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando ProteinBERT (local, ~/proteinbert_models/default.pkl)...", flush=True)
    pretrained_model_generator, input_encoder = load_pretrained_model(download_model_dump_if_not_exists=False)

    print(f"Cargando ESM-2 650M ({esm_model_name}, offline local)...", flush=True)
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    esm_model = EsmModel.from_pretrained(esm_model_name).eval()

    print(f"Cargando ProtT5 ({t5_model_path}, offline local)...", flush=True)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path, do_lower_case=False)
    t5_model = T5EncoderModel.from_pretrained(t5_model_path).eval()

    lines = dataset_path.read_text().splitlines()

    proteinbert_rows, esm_rows, t5_rows = [], [], []
    for i in range(0, len(lines), 2):
        header = lines[i].split(",")
        protein_id = header[0]
        positions = [int(p) for p in header[1:]]
        seq = lines[i + 1]

        print(f"[{protein_id}] {len(positions)} sitio(s) candidato(s), {len(seq)} aa", flush=True)
        pb_full = _get_proteinbert_representation(pretrained_model_generator, input_encoder, seq)
        esm_full = _get_esm2_embedding(esm_tokenizer, esm_model, seq)
        t5_full = _get_prott5_embedding(t5_tokenizer, t5_model, seq)

        for pos in positions:
            proteinbert_rows.append(pb_full)
            start = max(pos - _WINDOW_SIZE - 1, 0)
            end = min(pos + _WINDOW_SIZE, len(seq))
            esm_rows.append(np.mean(esm_full[start:end, :], axis=0))
            t5_rows.append(t5_full[pos - 1])

    features = np.concatenate([np.array(proteinbert_rows), np.array(esm_rows), np.array(t5_rows)], axis=1)
    features_path = output_dir / "features.csv"
    np.savetxt(features_path, features, delimiter=",")
    return features_path


def _preprocess(feature_x: np.ndarray, stage: int, models_dir: Path) -> np.ndarray:
    with open(models_dir / f"power_transformer_{stage}.sav", "rb") as f:
        pt = pickle.load(f)
    return pt.transform(feature_x)


def _base_layer_predictions(feature_x: np.ndarray, models_dir: Path) -> np.ndarray:
    test_x = _preprocess(feature_x, 2, models_dir)
    total = np.zeros((len(test_x), 1), dtype=float)
    pickle_dir = models_dir / "base_layer_pickle_files"

    for i in range(10):
        for base_classifier in ("SVM", "XGB", "KNN"):
            with open(pickle_dir / f"{base_classifier}_base_layer_{i}.sav", "rb") as f:
                model = pickle.load(f)
            y_proba = model.predict_proba(test_x)[:, 1].reshape(-1, 1)
            total = np.concatenate((total, y_proba), axis=1)

    return np.delete(total, 0, axis=1)


def predict(features_path: Path, output_dir: Path, models_dir: Path) -> Path:
    """Aplica el stack de clasificadores (base layer + meta-SVM) ya entrenados, sin modificaciones."""
    feature_x = np.loadtxt(features_path, delimiter=",")
    if feature_x.ndim == 1:
        feature_x = feature_x.reshape(1, -1)

    x = _preprocess(feature_x, 1, models_dir)
    blp = _base_layer_predictions(x, models_dir)
    x = np.concatenate((x, blp), axis=1)
    x = _preprocess(x, 3, models_dir)

    with open(models_dir / "base_layer_pickle_files" / "SVM_meta_layer.sav", "rb") as f:
        clf = pickle.load(f)
    y_pred = clf.predict(x)
    y_proba = clf.predict_proba(x)[:, 1]

    predicted_path = output_dir / "predicted_values.csv"
    np.savetxt(predicted_path, np.column_stack([y_pred, y_proba]), delimiter=",", fmt="%.6f",
               header="prediction,probability", comments="")
    return predicted_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="dataset.txt (formato original StackGlyEmbed)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Carpeta donde escribir features.csv y predicted_values.csv")
    parser.add_argument("--models-dir", required=True, type=Path,
                         help="Carpeta 'prediction/' del clon de StackGlyEmbed (power_transformer_*.sav + base_layer_pickle_files/)")
    parser.add_argument("--t5-model-path", default=_DEFAULT_T5_MODEL_PATH, help="Ruta local a los pesos de ProtT5")
    parser.add_argument("--esm-model-name", default=_DEFAULT_ESM_MODEL, help="ID de HF Hub del modelo ESM-2 (offline si ya esta cacheado)")
    args = parser.parse_args()

    features_path = extract_features(args.dataset, args.output_dir, args.t5_model_path, args.esm_model_name)
    predicted_path = predict(features_path, args.output_dir, args.models_dir)
    print(f"-> Predicciones guardadas en: {predicted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
