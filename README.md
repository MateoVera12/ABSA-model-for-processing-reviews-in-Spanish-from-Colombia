# ABSA for Colombian Spanish Reviews

**Aspect-Based Sentiment Analysis (ABSA)** for Spanish-language app reviews, focused on extracting **evidence spans, aspects, categories and sentiment** from real-world Colombian Spanish text.

> Research project + inference application developed as part of a Systems Engineering thesis.

## Why this project matters

Traditional sentiment analysis answers *whether* a review is positive or negative. This project goes further by identifying **what part of the review expresses the opinion** and **which aspect it refers to**.

Example concept:

```text
"La aplicación es rápida, pero el inicio de sesión falla mucho."

→ evidence span: "el inicio de sesión falla mucho"
→ aspect: authentication / login
→ sentiment: negative
```

## Features

- Span-level evidence extraction.
- Aspect and category prediction.
- Sentiment analysis over detected evidence.
- Training and evaluation pipeline.
- Optional K-Fold cross-validation.
- Batch inference over multiple reviews.
- Interactive **Streamlit** application for model inference and visualization.
- Model loading from **Hugging Face Hub**.
- Visual highlighting of detected evidence according to sentiment.
- Tabular exploration of predictions and distributions.

## Tech stack

| Area | Technologies |
| --- | --- |
| Language | Python 3.10+ |
| NLP / Deep Learning | PyTorch, Transformers, Hugging Face |
| Data | Pandas, NumPy, Hugging Face Datasets |
| ML evaluation | Scikit-learn |
| Application | Streamlit |
| Model distribution | Hugging Face Hub |
| Training | GPU/CUDA supported |

## Architecture

```text
                    ┌─────────────────────┐
                    │   Spanish Review    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Tokenization / NLP  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ABSA Model        │
                    │ Evidence + Aspects  │
                    │ + Category + Sent.  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Structured JSON │        │ Streamlit UI    │
        │ predictions     │        │ visualization   │
        └─────────────────┘        └─────────────────┘
```

## Repository structure

```text
.
├── absa_evidence_model.py   # Model, training and evaluation
├── absa_inference.py        # Checkpoint loading and inference
├── app.py                   # Streamlit application
├── evaluator.py             # Evaluation and reporting utilities
├── data/                    # Labeled datasets
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/MateoVera12/ABSA-model-for-processing-reviews-in-Spanish-from-Colombia.git
cd ABSA-model-for-processing-reviews-in-Spanish-from-Colombia
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA-enabled training, install the appropriate PyTorch build for your environment before installing the remaining dependencies if necessary.

## Training

Standard training/evaluation:

```bash
python absa_evidence_model.py --data "data/etiquetas/reseñas_etiquetadas_3493.json"
```

5-fold cross-validation:

```bash
python absa_evidence_model.py \
  --data "data/etiquetas/reseñas_etiquetadas_3493.json" \
  --cv_folds 5 \
  --cv_output_json "cv5_results.json"
```

To save a checkpoint for each fold:

```bash
python absa_evidence_model.py \
  --data "data/etiquetas/reseñas_etiquetadas_3493.json" \
  --cv_folds 5 \
  --save_fold_models
```

## Running the application

Start the Streamlit interface with:

```bash
streamlit run app.py
```

The application downloads the inference model from Hugging Face Hub when required, loads the checkpoint, and provides an interactive interface for analyzing reviews.

## Research context

The project was developed for research on **aspect-based sentiment analysis in Colombian Spanish app reviews**. The repository contains both the model-training pipeline and an application layer that demonstrates how the trained model can be used in a practical NLP workflow.

The combination of **NLP research, model training, evaluation and application development** makes this project representative of an end-to-end machine-learning workflow rather than a standalone notebook experiment.

## Author

**Carlos Mateo Vera Grimaldo**  
Systems Engineer · Backend / Software Development · NLP & AI

GitHub: [@MateoVera12](https://github.com/MateoVera12)
