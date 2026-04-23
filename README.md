# Evidence ABSA (Aspect-Based Sentiment Analysis)

Implementación del modelo de **ABSA span-level** para reseñas en español, junto con:

- Entrenamiento/evaluación (script de investigación).
- Página web en **Streamlit** para inferencia y visualización de evidencias.

## Requisitos

- Python 3.10+ (recomendado)
- (Opcional) GPU con CUDA para acelerar entrenamiento

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Nota: `torch` puede requerir una instalación específica según tu sistema/CUDA. Si tienes problemas instalándolo desde `requirements.txt`, instala PyTorch primero y luego vuelve a ejecutar el comando de instalación.

## Dataset (entrenamiento)

Por defecto el script apunta a:

```text
data/reseñas_etiquetadas_3493.json
```

El archivo debe ser un JSON con una lista de ejemplos etiquetados (formato usado en el trabajo de grado).

## Entrenamiento (modo estándar)

Entrena el modelo y evalúa sobre el *test set*. Al finalizar guarda el checkpoint:

- `evidence_absa_model.pt`

Comando:

```bash
python absa_evidence_model.py --data "data/etiquetas/reseñas_etiquetadas_3493.json"
```

## Entrenamiento con Cross-Validation (opcional)

Ejecuta K-Fold CV (ej. 5 folds) y guarda un resumen agregado en JSON:

```bash
python absa_evidence_model.py --data "data/etiquetas/reseñas_etiquetadas_3493.json" --cv_folds 5 --cv_output_json "cv5_results.json"
```

Si además quieres guardar un `.pt` por cada fold:

```bash
python absa_evidence_model.py --data "data/etiquetas/reseñas_etiquetadas_3493.json" --cv_folds 5 --save_fold_models
```

Esto genera archivos tipo:

- `evidence_absa_model_fold1.pt`, `evidence_absa_model_fold2.pt`, ...

## Página web (Streamlit)

La app está en `app.py` y usa `absa_inference.py` para cargar el checkpoint y correr inferencia.

Ejecutar localmente:

```bash
streamlit run app.py
```

### Modelo usado por la app

Por defecto, la app descarga el modelo desde Hugging Face Hub usando `huggingface_hub.hf_hub_download` (ver `get_model_path()` en `app.py`).

- `repo_id`: `MateoV12/ABSA_evidence_model`
- `filename`: `evidence_absa_model.pt.zip`


## Estructura del repo

- `absa_evidence_model.py`: modelo + entrenamiento/evaluación (modo estándar y CV)
- `absa_inference.py`: carga del checkpoint + inferencia sobre textos
- `app.py`: página web (Streamlit)
- `evaluator.py`: evaluación (spans/tripletas) y utilidades de reporte
- `data/`: dataset(s)

