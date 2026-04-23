#!/usr/bin/env python3
import json
import random
import os
import time
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

from evaluator import evaluate_dataset, format_duration, print_fold_summary, print_cv_summary

# ---------------------------- Config ----------------------------
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
LR = 2e-5
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 50

ASPECT_LABELS = [
    "autenticacion_ingreso",
    "gestion_cuenta",
    "transferencias_movimientos",
    "estabilidad",
    "notificaciones",
    "atencion_soporte",
    "interfaz",
    "flujo_uso",
    "contenido",
    "promociones_reembolsos",
    "app",
    "envio",
    "anuncios",
    "consumo_recursos"
]

CATEGORY_LABELS = [
    "funcionalidad",
    "usabilidad",
    "rendimiento",
    "seguridad",
    "costos",
    "general"
]

SENTIMENT_LABELS = ["POS", "NEU", "NEG"]  

# -------------------- Umbrales y settings para decoding --------------------
ASPECT_CONF_THRESH = 0.7
CAT_CONF_THRESH = 0.7

# Decoding spans
SPAN_TEMPERATURE = 1.5      # suaviza logits antes de sigmoid
START_THRESH = 0.35       
END_THRESH = 0.35
TOP_K_STARTS = 20
LENGTH_PENALTY_COEF = 0.04  # penaliza spans largos 
NMS_IOU_THRESH = 0.25   # evita que un span grande tape varios pequeños
MAX_CANDIDATES_PER_EXAMPLE = 300
ALLOW_OVERLAP = False

# ------------------ Reproducibilidad ------------------
def set_seed(seed: int = 42, verbose: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print(torch.__version__)
        print(torch.cuda.is_available())
    if torch.cuda.is_available():
        if verbose:
            print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------- Utils ----------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def span_char_iou(span_a, span_b):
    """
    IoU entre spans de caracteres [start, end).
    """
    a0, a1 = span_a
    b0, b1 = span_b

    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)

    if union <= 0:
        return 0.0
    return inter / union

def find_word_span_char_offsets(text: str, start_word: int, end_word: int) -> Tuple[int, int]:
    """
    Mapea índices por palabra (0-based, inclusive) a offsets de caracteres half-open [start, end).
    Usa text.split() y fallback por búsqueda del substring.
    """ 
    words = text.split()
    
    if start_word < 0 or end_word >= len(words) or start_word > end_word:
        print(f"Warning: Invalid word span {start_word}-{end_word} for text with {text} words. Attempting fallback.")
        raise ValueError(f"Invalid word span indices for text: {start_word}-{end_word} (words={text})")

    char_start = None
    char_end_inclusive = None
    current_word = 0
    i = 0
    n = len(text)

    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and (not text[j].isspace()):
            j += 1

        if current_word == start_word:
            char_start = i
        if current_word == end_word:
            char_end_inclusive = j - 1
            break

        current_word += 1
        i = j

    if char_start is None or char_end_inclusive is None:
        joined = " ".join(words[start_word:end_word + 1])
        found = text.find(joined)
        if found == -1:
            raise ValueError("Could not map word span to char offsets (fallback failed).")
        char_start = found
        char_end_inclusive = found + len(joined) - 1

    return char_start, char_end_inclusive + 1


# ---------------------------- Dataset ----------------------------
class ABSAEvidenceDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = item.get("label", [])

        processed = {
            "text": text,
            "labels": [],
            "neutral": False
        }

        if not labels:
            processed["neutral"] = True
            return processed

        for lab in labels:
            evid = lab.get("evidence_span", None)
            aspect = lab.get("aspect", [])
            category = lab.get("category", None)
            sentiment = lab.get("sentiment", None)

            if evid is None:
                raise ValueError("Todas las tripletas deben tener 'evidence_span.")

            if not aspect or len(aspect) != 1:
                raise ValueError("Cada etiqueta debe tener exactamente 1 aspecto válido. Found: " + str(aspect))
            aspect_name = aspect[0]
            if aspect_name not in ASPECT_LABELS:
                raise ValueError(f"Unknown aspect '{aspect_name}'. Add to ASPECT_LABELS or clean data.")
            aspect_idx = ASPECT_LABELS.index(aspect_name)

            if category not in CATEGORY_LABELS:
                raise ValueError(f"Unknown category '{category}'. Add to CATEGORY_LABELS or clean data.")
            cat_idx = CATEGORY_LABELS.index(category)

            if sentiment not in SENTIMENT_LABELS:
                raise ValueError(f"Unknown sentiment '{sentiment}'. Expected one of: {SENTIMENT_LABELS}")
            sent_idx = SENTIMENT_LABELS.index(sentiment)

            if not isinstance(evid, list) or len(evid) != 2:
                raise ValueError("evidence_span must be [start_word_idx, end_word_idx]. Got: " + str(evid))

            wstart, wend = evid
            char_start, char_end_excl = find_word_span_char_offsets(text, wstart, wend)

            processed["labels"].append({
                "char_span": (char_start, char_end_excl),
                "aspect_idx": aspect_idx,
                "cat_idx": cat_idx,
                "sent_idx": sent_idx
            })

        return processed

    def collate_fn(self, batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        batch_size, seq_len = enc["input_ids"].shape

        start_labels = torch.zeros((batch_size, seq_len), dtype=torch.float)
        end_labels = torch.zeros((batch_size, seq_len), dtype=torch.float)

        flat_aspect = []
        flat_cat = []
        flat_sent = []
        span_to_example = []

        offsets_batch = [
            [(int(a), int(b)) for (a, b) in enc["offset_mapping"][i].tolist()]
            for i in range(batch_size)
        ]

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        meta = []

        for i, item in enumerate(batch):
            meta.append({
                "text": item["text"],
                "neutral": bool(item.get("neutral", False)),
                "labels": item["labels"]
            })

            if not item.get("labels", None):
                continue

            for lab in item["labels"]:
                cstart, cend = lab["char_span"]
                token_offsets = offsets_batch[i]
                tok_start = None
                tok_end = None

                for t_idx, (ofs0, ofs1) in enumerate(token_offsets):
                    if ofs0 == 0 and ofs1 == 0:
                        continue
                    if tok_start is None and ofs1 > cstart:
                        tok_start = t_idx
                    if ofs0 < cend:
                        tok_end = t_idx

                if tok_start is None or tok_end is None or tok_start > tok_end:
                    continue

                start_labels[i, tok_start] = 1.0
                end_labels[i, tok_end] = 1.0

                flat_aspect.append(lab["aspect_idx"])
                flat_cat.append(lab["cat_idx"])
                flat_sent.append(lab["sent_idx"])
                span_to_example.append((i, tok_start, tok_end))

        has_spans = len(flat_aspect) > 0

        if has_spans:
            flat_aspect_idx = torch.tensor(flat_aspect, dtype=torch.long)
            flat_cat_idx = torch.tensor(flat_cat, dtype=torch.long)

            flat_aspect_oh = torch.zeros((len(flat_aspect), len(ASPECT_LABELS)), dtype=torch.float)
            flat_cat_oh = torch.zeros((len(flat_cat), len(CATEGORY_LABELS)), dtype=torch.float)

            for r, a_idx in enumerate(flat_aspect_idx.tolist()):
                flat_aspect_oh[r, a_idx] = 1.0
            for r, c_idx in enumerate(flat_cat_idx.tolist()):
                flat_cat_oh[r, c_idx] = 1.0

            flat_sent_t = torch.tensor(flat_sent, dtype=torch.long)
        else:
            flat_aspect_idx = None
            flat_cat_idx = None
            flat_aspect_oh = None
            flat_cat_oh = None
            flat_sent_t = None
            span_to_example = []

        batch_output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_labels": start_labels,
            "end_labels": end_labels,
            "span_to_example": span_to_example,
            "flat_aspect": flat_aspect_oh,
            "flat_cat": flat_cat_oh,
            "flat_aspect_idx": flat_aspect_idx,
            "flat_cat_idx": flat_cat_idx,
            "flat_sent": flat_sent_t,
            "offset_mapping": offsets_batch,
            "meta": meta
        }

        return batch_output


# ---------------------------- Model ----------------------------
class EvidenceABSA(nn.Module):
    def __init__(self, model_name: str, hidden_dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        self.start_classifier = nn.Linear(d_model, 1)
        self.end_classifier = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(hidden_dropout)

        fusion_dim = d_model * 2

        self.aspect_classifier = nn.Linear(fusion_dim, len(ASPECT_LABELS))
        self.cat_classifier = nn.Linear(fusion_dim + d_model, len(CATEGORY_LABELS))

        self.aspect_emb = nn.Embedding(len(ASPECT_LABELS), d_model)
        self.cat_emb = nn.Embedding(len(CATEGORY_LABELS), d_model)

        self.sent_classifier = nn.Linear(d_model * 3, len(SENTIMENT_LABELS))

        nn.init.xavier_uniform_(self.aspect_emb.weight)
        nn.init.xavier_uniform_(self.cat_emb.weight)

    def forward(self, input_ids, attention_mask, span_to_example: List[Tuple[int, int, int]] = None, target_aspect_idx=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else hidden[:, 0, :]

        start_logits = self.start_classifier(self.dropout(hidden)).squeeze(-1)
        end_logits = self.end_classifier(self.dropout(hidden)).squeeze(-1)

        results = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden": hidden,
            "pooled": pooled
        }

        if span_to_example is None or len(span_to_example) == 0:
            results.update({
                "aspect_logits": None,
                "cat_logits": None,
                "span_embeddings": None,
                "span_fusions": None
            })
            return results

        span_embs = []
        span_fusions = []

        for (ex_i, s_tok, e_tok) in span_to_example:
            span_hidden = hidden[ex_i, s_tok:(e_tok + 1), :]
            if span_hidden.size(0) == 0:
                span_mean = torch.zeros(hidden.size(-1), device=hidden.device)
            else:
                span_mean = span_hidden.mean(dim=0)

            cls_emb = pooled[ex_i]
            fusion = torch.cat([span_mean, cls_emb], dim=-1)

            span_embs.append(span_mean)
            span_fusions.append(fusion)

        span_embeddings = torch.stack(span_embs, dim=0)
        span_fusions = torch.stack(span_fusions, dim=0)

        aspect_logits = self.aspect_classifier(self.dropout(span_fusions))

        if target_aspect_idx is not None:
            asp_embs = self.aspect_emb(target_aspect_idx)
            cat_input = torch.cat([span_fusions, asp_embs], dim=-1)
            cat_logits = self.cat_classifier(self.dropout(cat_input))
        else:
            cat_logits = None

        results.update({
            "aspect_logits": aspect_logits,
            "cat_logits": cat_logits,
            "span_embeddings": span_embeddings,
            "span_fusions": span_fusions
        })
        return results


# ---------------------------- Loss & Training helpers ----------------------------
def compute_loss(model, batch, outputs, loss_fns, device):
    """
    Loss:
     - start/end token BCE
     - aspect BCE multi-label
     - category BCE multi-label
     - sentiment CE
    """
    loss = 0.0
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]

    start_labels = batch["start_labels"].to(device)
    end_labels = batch["end_labels"].to(device)

    span_loss_fn = loss_fns["span"]
    span_loss = span_loss_fn(start_logits, start_labels) + span_loss_fn(end_logits, end_labels)
    loss += span_loss

    if batch.get("flat_aspect", None) is None:
        logs = {"span_loss": float(span_loss.detach().cpu().item())}
        return loss, logs

    aspect_logits = outputs["aspect_logits"]
    cat_logits = outputs["cat_logits"]
    span_embeddings = outputs["span_embeddings"]

    aspect_targets = batch["flat_aspect"].to(device)
    cat_targets = batch["flat_cat"].to(device)
    sent_targets = batch["flat_sent"].to(device)

    aspect_idx = batch.get("flat_aspect_idx", None)
    cat_idx = batch.get("flat_cat_idx", None)

    if cat_logits is None:
        raise RuntimeError("cat_logits is None. Ensure 'target_aspect_idx' is passed to model forward in compute_loss.")

    loss += loss_fns["aspect"](aspect_logits, aspect_targets)
    loss += loss_fns["cat"](cat_logits, cat_targets)

    if aspect_idx is None or cat_idx is None:
        raise RuntimeError("flat_aspect_idx and flat_cat_idx must be present for sentiment conditioning during training.")

    aspect_idx = aspect_idx.to(device) if torch.is_tensor(aspect_idx) else torch.tensor(aspect_idx, dtype=torch.long, device=device)
    cat_idx = cat_idx.to(device) if torch.is_tensor(cat_idx) else torch.tensor(cat_idx, dtype=torch.long, device=device)

    asp_embs = model.aspect_emb(aspect_idx)
    cat_embs = model.cat_emb(cat_idx)

    rel_input = torch.cat([span_embeddings, asp_embs, cat_embs], dim=-1)
    rel_input = model.dropout(rel_input)
    sent_logits = model.sent_classifier(rel_input)

    loss += loss_fns["sent"](sent_logits, sent_targets)

    logs = {
        "span_loss": float(span_loss.detach().cpu().item())
    }
    return loss, logs


def train_epoch(model, dataloader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        target_aspect_idx = batch.get("flat_aspect_idx", None)
        if target_aspect_idx is not None:
            target_aspect_idx = target_aspect_idx.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_to_example=batch["span_to_example"],
            target_aspect_idx=target_aspect_idx
        )

        loss, logs = compute_loss(model, batch, outputs, loss_fns, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % PRINT_EVERY == 0:
            pbar.set_description(f"loss:{loss.item():.4f}")

    return total_loss / len(dataloader)


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return data_copy[:train_end], data_copy[train_end:val_end], data_copy[val_end:]


def split_train_val(data, val_ratio=0.1, seed=42):
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    n = len(data_copy)
    val_size = int(n * val_ratio)
    if n > 1:
        val_size = max(1, val_size)
        val_size = min(val_size, n - 1)
    else:
        val_size = 0
    return data_copy[val_size:], data_copy[:val_size]


def build_dataloaders(tokenizer, train_data, val_data, test_data, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    train_ds = ABSAEvidenceDataset(train_data, tokenizer, max_len=max_len)
    val_ds = ABSAEvidenceDataset(val_data, tokenizer, max_len=max_len)
    test_ds = ABSAEvidenceDataset(test_data, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_ds.collate_fn)
    return train_loader, val_loader, test_loader


def build_training_components(model_name: str):
    model = EvidenceABSA(model_name=model_name).to(DEVICE)

    loss_fns = {
        "span": nn.BCEWithLogitsLoss(),
        "aspect": nn.BCEWithLogitsLoss(),
        "cat": nn.BCEWithLogitsLoss(),
        "sent": nn.CrossEntropyLoss(),
    }
    optimizer = AdamW(model.parameters(), lr=LR)
    return model, optimizer, loss_fns


def train_model(model, train_loader, val_loader, optimizer, loss_fns, device, num_epochs=NUM_EPOCHS, fold_label=None):
    header = f"Starting training for {fold_label}..." if fold_label else "Starting training..."
    print(header)
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        val_loss = evaluate_loss(model, val_loader, loss_fns, device)
        prefix = f"[{fold_label}] " if fold_label else ""
        print(
            f"{prefix}Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}"
        )
    return model


NUMERIC_TYPES = (int, float, np.integer, np.floating)


def _extract_per_class_metrics(per_class_dict, allowed_keys=None):
    extracted = {}
    for cls, stats in per_class_dict.items():
        if allowed_keys is not None and cls not in allowed_keys:
            continue
        extracted[cls] = {
            "support": float(stats.get("support", 0.0)),
            "prec": float(stats.get("prec", 0.0)),
            "rec": float(stats.get("rec", 0.0)),
            "f1": float(stats.get("f1", 0.0)),
        }
    return extracted


def _extract_classification_metrics(metric_block, allowed_per_class_keys=None):
    metric_block = metric_block or {}
    macro = metric_block.get("macro", {})
    return {
        "accuracy_given_span": float(metric_block.get("accuracy_given_span", 0.0)),
        "micro": {
            "prec": float(metric_block.get("micro", {}).get("prec", 0.0)),
            "rec": float(metric_block.get("micro", {}).get("rec", 0.0)),
            "f1": float(metric_block.get("micro", {}).get("f1", 0.0)),
        },
        "macro": {
            "prec": float(macro.get("prec", 0.0)),
            "rec": float(macro.get("rec", 0.0)),
            "f1": float(macro.get("f1", 0.0)),
            "weighted_prec": float(macro.get("weighted_prec", 0.0)),
            "weighted_rec": float(macro.get("weighted_rec", 0.0)),
            "weighted_f1": float(macro.get("weighted_f1", 0.0)),
            "per_class": _extract_per_class_metrics(
                macro.get("per_class", {}),
                allowed_keys=allowed_per_class_keys,
            ),
        },
    }


def extract_fold_metrics(metrics):
    return {
        "span_iou": {
            "predicted": float(metrics["span_iou"].get("predicted", 0.0)),
            "gold": float(metrics["span_iou"].get("gold", 0.0)),
            "tp": float(metrics["span_iou"].get("tp", 0.0)),
            "fp": float(metrics["span_iou"].get("fp", 0.0)),
            "fn": float(metrics["span_iou"].get("fn", 0.0)),
            "precision": float(metrics["span_iou"].get("precision", 0.0)),
            "recall": float(metrics["span_iou"].get("recall", 0.0)),
            "f1": float(metrics["span_iou"].get("f1", 0.0)),
        },
        "span_exact": {
            "predicted": float(metrics["span_exact"].get("predicted", 0.0)),
            "gold": float(metrics["span_exact"].get("gold", 0.0)),
            "tp": float(metrics["span_exact"].get("tp", 0.0)),
            "fp": float(metrics["span_exact"].get("fp", 0.0)),
            "fn": float(metrics["span_exact"].get("fn", 0.0)),
            "precision": float(metrics["span_exact"].get("precision", 0.0)),
            "recall": float(metrics["span_exact"].get("recall", 0.0)),
            "f1": float(metrics["span_exact"].get("f1", 0.0)),
        },
        "triplet_strict": {
            "tp": float(metrics["triplet_strict"].get("tp", 0.0)),
            "fp": float(metrics["triplet_strict"].get("fp", 0.0)),
            "fn": float(metrics["triplet_strict"].get("fn", 0.0)),
            "precision": float(metrics["triplet_strict"].get("precision", 0.0)),
            "recall": float(metrics["triplet_strict"].get("recall", 0.0)),
            "f1": float(metrics["triplet_strict"].get("f1", 0.0)),
        },
        "aspect": _extract_classification_metrics(metrics["aspect"]),
        "category": _extract_classification_metrics(metrics["category"]),
        "sentiment": _extract_classification_metrics(metrics["sentiment"]),
        "pair_cat_sent": _extract_classification_metrics(metrics["pair_cat_sent"]),
    }


def _aggregate_nested_metrics(metric_dicts):
    if not metric_dicts:
        return {}, {}

    all_keys = set()
    for item in metric_dicts:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    mean = {}
    std = {}
    for key in sorted(all_keys):
        values = [item.get(key) if isinstance(item, dict) else None for item in metric_dicts]
        sample = next((v for v in values if isinstance(v, dict) or isinstance(v, NUMERIC_TYPES)), None)
        if sample is None:
            continue

        if isinstance(sample, dict):
            sub_mean, sub_std = _aggregate_nested_metrics(
                [v if isinstance(v, dict) else {} for v in values]
            )
            mean[key] = sub_mean
            std[key] = sub_std
            continue

        numeric_values = [
            float(v) if isinstance(v, NUMERIC_TYPES) else 0.0
            for v in values
        ]
        mean[key] = float(np.mean(numeric_values))
        std[key] = float(np.std(numeric_values))

    return mean, std


def aggregate_fold_summaries(fold_summaries):
    if not fold_summaries:
        return {"mean": {}, "std": {}}

    metric_dicts = [fold["metrics"] for fold in fold_summaries]
    mean, std = _aggregate_nested_metrics(metric_dicts)
    return {"mean": mean, "std": std}


def evaluate_loss(model, dataloader, loss_fns, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            target_aspect_idx = batch.get("flat_aspect_idx", None)
            if target_aspect_idx is not None:
                target_aspect_idx = target_aspect_idx.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_to_example=batch["span_to_example"],
                target_aspect_idx=target_aspect_idx
            )
            loss, _ = compute_loss(model, batch, outputs, loss_fns, device)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------- Decoding & Filtering ----------------------------
def decode_spans_joint(
    start_logits,
    end_logits,
    hidden_states,
    offset_mapping,
    start_thresh=START_THRESH,
    end_thresh=END_THRESH,
    top_k_starts=TOP_K_STARTS,
    allow_overlap=ALLOW_OVERLAP,
    temperature=SPAN_TEMPERATURE,
    length_penalty_coef=LENGTH_PENALTY_COEF,
    nms_iou_thresh=NMS_IOU_THRESH,
    max_candidates=MAX_CANDIDATES_PER_EXAMPLE
):
    start_probs = torch.sigmoid(start_logits / temperature)
    end_probs = torch.sigmoid(end_logits / temperature)

    H = hidden_states
    seq_len = H.size(0)

    valid_starts = [
        i for i in range(seq_len)
        if offset_mapping[i] != (0, 0) and float(start_probs[i].item()) >= start_thresh
    ]

    if not valid_starts:
        return []

    valid_starts = sorted(
        valid_starts,
        key=lambda i: float(start_probs[i].item()),
        reverse=True
    )[:top_k_starts]

    candidates = []

    for s in valid_starts:
        h_s = H[s]

        for e in range(s, seq_len):
            if offset_mapping[e] == (0, 0):
                continue
            if float(end_probs[e].item()) < end_thresh:
                continue

            h_e = H[e]

            # Similitud contextual estable en [0, 1]
            sim = F.cosine_similarity(
                h_s.unsqueeze(0),
                h_e.unsqueeze(0),
                dim=-1
            ).item()
            sim = (sim + 1.0) / 2.0

            # Penalización suave por longitud en tokens
            span_len_tokens = (e - s + 1)
            length_penalty = 1.0 / (1.0 + length_penalty_coef * max(0, span_len_tokens - 1))

            # Score 
            base_score = (float(start_probs[s].item()) + float(end_probs[e].item())) / 2.0
            score = base_score * sim * length_penalty

            char_start = int(offset_mapping[s][0])
            char_end = int(offset_mapping[e][1])

            if char_end <= char_start:
                continue
            

            candidates.append({
                "s_tok": s,
                "e_tok": e,
                "start_prob": float(start_probs[s].item()),
                "end_prob": float(end_probs[e].item()),
                "score": float(score),
                "char_start": char_start,
                "char_end": char_end
            })

    if not candidates:
        return []

    # Quedarse con los mejores candidatos antes del NMS
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    if allow_overlap:
        return [(c["char_start"], c["char_end"]) for c in candidates]

    # NMS: conserva spans no muy solapados, priorizando el score más alto
    selected = []
    for c in candidates:
        c_span = (c["char_start"], c["char_end"])
        suppress = False

        for d in selected:
            d_span = (d["char_start"], d["char_end"])
            if span_char_iou(c_span, d_span) > nms_iou_thresh:
                suppress = True
                break

        if not suppress:
            selected.append(c)

    # Orden final por posición para devolver algo más legible
    selected = sorted(selected, key=lambda x: (x["char_start"], x["char_end"]))

    return [(c["char_start"], c["char_end"]) for c in selected]

def decode_batch(model, batch):
    """
    Decoding: no neutral gating; all tripletas requieren span.
    Returns list per example:
    {"neutral": bool, "evidences": [ {char_span, aspect, category, sentiment}, ... ]}
    """
    model.eval()

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    offsets_batch = batch["offset_mapping"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    hidden = outputs["hidden"]
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]
    pooled = outputs["pooled"]

    batch_size = input_ids.size(0)
    results = []

    for i in range(batch_size):
        offsets = offsets_batch[i]

        spans = decode_spans_joint(
            start_logits[i],
            end_logits[i],
            hidden[i],
            offsets,
            start_thresh=START_THRESH,
            end_thresh=END_THRESH,
            top_k_starts=TOP_K_STARTS,
            allow_overlap=ALLOW_OVERLAP,
            temperature=SPAN_TEMPERATURE,
            length_penalty_coef=LENGTH_PENALTY_COEF,
            nms_iou_thresh=NMS_IOU_THRESH,
            max_candidates=MAX_CANDIDATES_PER_EXAMPLE
        )

        evidences = []

        for (char_start, char_end) in spans:
            tok_start = None
            tok_end = None

            for t_idx, (ofs0, ofs1) in enumerate(offsets):
                if ofs0 == 0 and ofs1 == 0:
                    continue
                if tok_start is None and ofs1 > char_start:
                    tok_start = t_idx
                if ofs0 < char_end:
                    tok_end = t_idx

            if tok_start is None or tok_end is None or tok_end < tok_start:
                continue

            span_hidden = hidden[i, tok_start:(tok_end + 1), :]
            if span_hidden.size(0) == 0:
                continue

            span_emb = span_hidden.mean(dim=0, keepdim=True)
            cls_emb = pooled[i].unsqueeze(0)
            fusion = torch.cat([span_emb, cls_emb], dim=-1)
            fusion = model.dropout(fusion)

            # Step 1: aspect
            aspect_logits = model.aspect_classifier(fusion).squeeze(0)
            aspect_probs = torch.sigmoid(aspect_logits).detach().cpu().numpy()
            selected_aspect_idxs = list(np.where(aspect_probs >= ASPECT_CONF_THRESH)[0])

            if len(selected_aspect_idxs) == 0:
                selected_aspect_idxs = [int(torch.argmax(aspect_logits).item())]

            for a_idx in selected_aspect_idxs:
                # Step 2: category conditioned on aspect
                asp_emb = model.aspect_emb(torch.tensor(a_idx, device=DEVICE)).unsqueeze(0)
                cat_input = torch.cat([fusion, asp_emb], dim=-1)
                cat_logits = model.cat_classifier(cat_input).squeeze(0)
                cat_probs = torch.sigmoid(cat_logits).detach().cpu().numpy()
                selected_cat_idxs = list(np.where(cat_probs >= CAT_CONF_THRESH)[0])

                if len(selected_cat_idxs) == 0:
                    selected_cat_idxs = [int(torch.argmax(cat_logits).item())]

                for c_idx in selected_cat_idxs:
                    # Step 3: sentiment conditioned on aspect and category
                    cat_emb = model.cat_emb(torch.tensor(c_idx, device=DEVICE)).unsqueeze(0)
                    rel = torch.cat([span_emb, asp_emb, cat_emb], dim=-1)
                    rel = model.dropout(rel)
                    sent_logits_pair = model.sent_classifier(rel)
                    sent_idx = int(torch.argmax(sent_logits_pair, dim=-1).item())

                    evidences.append({
                        "char_span": (int(char_start), int(char_end)),
                        "aspect": ASPECT_LABELS[a_idx],
                        "category": CATEGORY_LABELS[c_idx],
                        "sentiment": SENTIMENT_LABELS[sent_idx]
                    })

        neutral_flag = (len(evidences) == 0)
        results.append({
            "neutral": neutral_flag,
            "evidences": evidences
        })

    return results


# ---------------------------- Build gold & evaluate ----------------------------
def build_gold(batch):
    golds = []
    meta = batch.get("meta", None)
    if meta is None:
        raise RuntimeError("No meta in batch; collate must provide 'meta'.")

    for i in range(len(meta)):
        m = meta[i]
        gold_evids = []
        for lab in m["labels"]:
            gold_evids.append({
                "char_span": lab["char_span"],
                "aspect": ASPECT_LABELS[lab["aspect_idx"]],
                "category": CATEGORY_LABELS[lab["cat_idx"]],
                "sentiment": SENTIMENT_LABELS[lab["sent_idx"]]
            })
        golds.append({
            "text": m.get("text", None),
            "neutral": m.get("neutral", False),
            "evidences": gold_evids
        })
    return golds


def evaluate_absa(model, dataloader, verbose=True, show_examples=15):
    all_preds = []
    all_golds = []
    for batch in dataloader:
        batch_on_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_on_device[k] = v.to(DEVICE)
            else:
                batch_on_device[k] = v

        preds = decode_batch(model, batch_on_device)
        golds = build_gold(batch_on_device)
        all_preds.extend(preds)
        all_golds.extend(golds)

    metrics = evaluate_dataset(
        all_preds,
        all_golds,
        iou_threshold=0.5,
        exact_match_threshold=1.0,
        verbose=verbose,
        show_examples=show_examples
    )
    return metrics


def run_standard_training(data, tokenizer, model_name: str = MODEL_NAME):
    train_data, val_data, test_data = split_dataset(data)
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer,
        train_data,
        val_data,
        test_data,
    )

    model, optimizer, loss_fns = build_training_components(model_name)
    train_model(model, train_loader, val_loader, optimizer, loss_fns, DEVICE)

    print("Evaluating on test set...")
    metrics = evaluate_absa(model, test_loader, verbose=True, show_examples=15)

    out_path = "evidence_absa_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": model_name,
            "aspects": ASPECT_LABELS,
            "categories": CATEGORY_LABELS,
            "sentiments": SENTIMENT_LABELS
        }
    }, out_path)

    print("Model saved to", out_path)
    return metrics


def run_kfold_cross_validation(
    data,
    tokenizer,
    model_name: str = MODEL_NAME,
    cv_folds: int = 5,
    inner_val_ratio: float = 0.1,
    save_fold_models: bool = False,
    cv_output_json: str = "cv5_results.json",
):
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2 for cross-validation.")
    if not (0.0 < inner_val_ratio < 1.0):
        raise ValueError("inner_val_ratio must be between 0 and 1.")

    print(f"Running {cv_folds}-fold cross-validation...")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    indices = np.arange(len(data))
    fold_summaries = []

    for fold_idx, (train_pool_idx, test_idx) in enumerate(kf.split(indices), start=1):
        fold_seed = 42 + fold_idx
        set_seed(fold_seed, verbose=False)

        train_pool = [data[int(i)] for i in train_pool_idx]
        test_data = [data[int(i)] for i in test_idx]
        train_data, val_data = split_train_val(train_pool, val_ratio=inner_val_ratio, seed=fold_seed)

        train_loader, val_loader, test_loader = build_dataloaders(
            tokenizer,
            train_data,
            val_data,
            test_data,
        )

        model, optimizer, loss_fns = build_training_components(model_name)
        fold_label = f"fold {fold_idx}/{cv_folds}"
        fold_train_start = time.perf_counter()
        train_model(model, train_loader, val_loader, optimizer, loss_fns, DEVICE, fold_label=fold_label)
        fold_train_time_seconds = time.perf_counter() - fold_train_start

        print(f"Evaluating {fold_label} on held-out fold...")
        metrics = evaluate_absa(model, test_loader, verbose=False, show_examples=0)
        summary = extract_fold_metrics(metrics)
        print_fold_summary(
            fold_idx,
            cv_folds,
            len(train_data),
            len(val_data),
            len(test_data),
            summary,
            train_time_seconds=fold_train_time_seconds,
        )

        fold_entry = {
            "fold": fold_idx,
            "seed": fold_seed,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "train_time_seconds": float(fold_train_time_seconds),
            "train_time_human": format_duration(fold_train_time_seconds),
            "metrics": summary,
        }
        fold_summaries.append(fold_entry)

        if save_fold_models:
            fold_path = f"evidence_absa_model_fold{fold_idx}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_name": model_name,
                    "aspects": ASPECT_LABELS,
                    "categories": CATEGORY_LABELS,
                    "sentiments": SENTIMENT_LABELS,
                    "fold": fold_idx,
                }
            }, fold_path)
            print(f"Saved fold model to {fold_path}")

    aggregate = aggregate_fold_summaries(fold_summaries)
    print_cv_summary(aggregate)

    output = {
        "config": {
            "model_name": model_name,
            "cv_folds": cv_folds,
            "inner_val_ratio": inner_val_ratio,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "learning_rate": LR,
            "num_epochs": NUM_EPOCHS,
            "seed": 42,
        },
        "fold_metrics": fold_summaries,
        "aggregate": aggregate,
    }
    with open(cv_output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved CV summary to {cv_output_json}")
    return output


# ---------------------------- Main routine ----------------------------
def main(
    data_path: str,
    model_name: str = MODEL_NAME,
    cv_folds: int = 0,
    inner_val_ratio: float = 0.1,
    save_fold_models: bool = False,
    cv_output_json: str = "cv5_results.json",
):
    set_seed(42)
    print("Loading tokenizer and model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = load_json(data_path)

    if cv_folds and cv_folds > 1:
        return run_kfold_cross_validation(
            data,
            tokenizer,
            model_name=model_name,
            cv_folds=cv_folds,
            inner_val_ratio=inner_val_ratio,
            save_fold_models=save_fold_models,
            cv_output_json=cv_output_json,
        )

    return run_standard_training(data, tokenizer, model_name=model_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/reseñas_etiquetadas_3493.json",
        help="Path to dataset JSON"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="Number of folds for cross-validation. Use 5 to enable 5-fold CV."
    )
    parser.add_argument(
        "--inner_val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio extracted from each training pool during CV."
    )
    parser.add_argument(
        "--save_fold_models",
        action="store_true",
        help="Save one checkpoint per fold during cross-validation."
    )
    parser.add_argument(
        "--cv_output_json",
        type=str,
        default="cv5_results.json",
        help="Path to save the aggregated cross-validation JSON summary."
    )
    args = parser.parse_args()
    main(
        args.data,
        args.model_name,
        cv_folds=args.cv_folds,
        inner_val_ratio=args.inner_val_ratio,
        save_fold_models=args.save_fold_models,
        cv_output_json=args.cv_output_json,
    )
