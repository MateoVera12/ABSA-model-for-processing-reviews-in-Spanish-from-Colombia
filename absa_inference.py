import hashlib
import tempfile
import zipfile
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Default labels (fallback if checkpoint config is missing)
DEFAULT_ASPECT_LABELS = [
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
    "consumo_recursos",
]
DEFAULT_CATEGORY_LABELS = [
    "funcionalidad",
    "usabilidad",
    "rendimiento",
    "seguridad",
    "costos",
    "general",
]
DEFAULT_SENTIMENT_LABELS = ["POS", "NEU", "NEG"]

DEFAULT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
MIN_CHAR_LEN = 3

# Decoding defaults aligned with v11
SPAN_TEMPERATURE = 1.5
START_THRESH = 0.35
END_THRESH = 0.35
TOP_K_STARTS = 20
LENGTH_PENALTY_COEF = 0.04
NMS_IOU_THRESH = 0.35
MAX_CANDIDATES_PER_EXAMPLE = 300
ALLOW_OVERLAP = False

ASPECT_CONF_THRESH = 0.6
CAT_CONF_THRESH = 0.6


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: AutoTokenizer
    aspects: List[str]
    categories: List[str]
    sentiments: List[str]
    device: torch.device


class EvidenceABSA(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_aspects: int,
        n_categories: int,
        n_sentiments: int,
        hidden_dropout: float = 0.1,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size

        self.start_classifier = nn.Linear(d_model, 1)
        self.end_classifier = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(hidden_dropout)

        fusion_dim = d_model * 2
        self.aspect_classifier = nn.Linear(fusion_dim, n_aspects)
        self.cat_classifier = nn.Linear(fusion_dim + d_model, n_categories)

        self.aspect_emb = nn.Embedding(n_aspects, d_model)
        self.cat_emb = nn.Embedding(n_categories, d_model)

        self.sent_classifier = nn.Linear(d_model * 3, n_sentiments)

        nn.init.xavier_uniform_(self.aspect_emb.weight)
        nn.init.xavier_uniform_(self.cat_emb.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else hidden[:, 0, :]

        start_logits = self.start_classifier(self.dropout(hidden)).squeeze(-1)
        end_logits = self.end_classifier(self.dropout(hidden)).squeeze(-1)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden": hidden,
            "pooled": pooled,
        }


def _load_checkpoint(model_path: str) -> Dict:
    if os.path.isdir(model_path):
        zip_candidate = f"{model_path}.zip"
        if os.path.isfile(zip_candidate):
            model_path = zip_candidate
            return torch.load(model_path, map_location="cpu")

        parent = os.path.dirname(model_path)
        base = os.path.basename(model_path)
        sibling_zip = os.path.join(parent, f"{base}.zip")
        if os.path.isfile(sibling_zip):
            return torch.load(sibling_zip, map_location="cpu")

        inner_dir = None
        data_pkl = os.path.join(model_path, base, "data.pkl")
        data_dir = os.path.join(model_path, base, "data")
        if os.path.isfile(data_pkl) and os.path.isdir(data_dir):
            inner_dir = os.path.join(model_path, base)
        else:
            for entry in os.listdir(model_path):
                candidate = os.path.join(model_path, entry)
                if not os.path.isdir(candidate):
                    continue
                if os.path.isfile(os.path.join(candidate, "data.pkl")) and os.path.isdir(os.path.join(candidate, "data")):
                    inner_dir = candidate
                    break

        if inner_dir is not None:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(tmp_fd)
            try:
                with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(inner_dir):
                        for fname in files:
                            full_path = os.path.join(root, fname)
                            rel_path = os.path.relpath(full_path, model_path)
                            zf.write(full_path, rel_path)
                return torch.load(tmp_path, map_location="cpu")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        raise RuntimeError(
            "La ruta del modelo es un directorio. Proporciona un archivo .pt o .pt.zip."
        )
    return torch.load(model_path, map_location="cpu")


def _load_encoder_and_tokenizer(model_name: str):
    try:
        encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=True,
        )
        return encoder, tokenizer
    except OSError:
        pass

    try:
        encoder = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return encoder, tokenizer
    except OSError as exc:
        raise RuntimeError(
            "No se pudo cargar el encoder base "
            f"'{model_name}'. Verifica que esté en caché local o que la red permita descargarlo."
        ) from exc


def load_model_bundle(model_path: str) -> ModelBundle:
    ckpt = _load_checkpoint(model_path)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model_name = config.get("model_name", DEFAULT_MODEL_NAME)
    aspects = list(config.get("aspects", DEFAULT_ASPECT_LABELS))
    categories = list(config.get("categories", DEFAULT_CATEGORY_LABELS))
    sentiments = list(config.get("sentiments", DEFAULT_SENTIMENT_LABELS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, tokenizer = _load_encoder_and_tokenizer(model_name)
    model = EvidenceABSA(
        model_name,
        len(aspects),
        len(categories),
        len(sentiments),
        encoder=encoder,
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        raise RuntimeError("El checkpoint no contiene 'model_state_dict'.")

    model.to(device)
    model.eval()

    return ModelBundle(model=model, tokenizer=tokenizer, aspects=aspects, categories=categories, sentiments=sentiments, device=device)


def _prepare_batch(tokenizer, texts: List[str], max_len: int) -> Dict:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    batch_size = enc["input_ids"].shape[0]
    offsets_batch = [
        [(int(a), int(b)) for (a, b) in enc["offset_mapping"][i].tolist()]
        for i in range(batch_size)
    ]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offset_mapping": offsets_batch,
    }


def span_char_iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def decode_spans_joint(
    start_logits,
    end_logits,
    hidden_states,
    offset_mapping,
    start_thresh: float,
    end_thresh: float,
    top_k_starts: int,
    min_char_len: int,
    allow_overlap: bool,
    temperature: float,
    length_penalty_coef: float,
    nms_iou_thresh: float,
    max_candidates: int,
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
            sim = F.cosine_similarity(h_s.unsqueeze(0), h_e.unsqueeze(0), dim=-1).item()
            sim = (sim + 1.0) / 2.0

            span_len_tokens = (e - s + 1)
            length_penalty = 1.0 / (1.0 + length_penalty_coef * max(0, span_len_tokens - 1))

            base_score = (float(start_probs[s].item()) + float(end_probs[e].item())) / 2.0
            score = base_score * sim * length_penalty

            char_start = int(offset_mapping[s][0])
            char_end = int(offset_mapping[e][1])

            if char_end <= char_start:
                continue
            if (char_end - char_start) < min_char_len:
                continue

            candidates.append(
                {
                    "s_tok": s,
                    "e_tok": e,
                    "start_prob": float(start_probs[s].item()),
                    "end_prob": float(end_probs[e].item()),
                    "score": float(score),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    if allow_overlap:
        return [(c["char_start"], c["char_end"]) for c in candidates]

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

    selected = sorted(selected, key=lambda x: (x["char_start"], x["char_end"]))
    return [(c["char_start"], c["char_end"]) for c in selected]


def _decode_batch(
    bundle: ModelBundle,
    batch: Dict,
    start_thresh: float,
    end_thresh: float,
    top_k_starts: int,
    allow_overlap: bool,
    temperature: float,
    length_penalty_coef: float,
    nms_iou_thresh: float,
    max_candidates: int,
    aspect_conf_thresh: float,
    cat_conf_thresh: float,
):
    model = bundle.model
    device = bundle.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
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
            start_thresh=start_thresh,
            end_thresh=end_thresh,
            top_k_starts=top_k_starts,
            min_char_len=MIN_CHAR_LEN,
            allow_overlap=allow_overlap,
            temperature=temperature,
            length_penalty_coef=length_penalty_coef,
            nms_iou_thresh=nms_iou_thresh,
            max_candidates=max_candidates,
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

            span_hidden = hidden[i, tok_start : (tok_end + 1), :]
            if span_hidden.size(0) == 0:
                continue
            span_emb = span_hidden.mean(dim=0, keepdim=True)
            cls_emb = pooled[i].unsqueeze(0)
            fusion = torch.cat([span_emb, cls_emb], dim=-1)

            fusion = model.dropout(fusion)
            aspect_logits = model.aspect_classifier(fusion).squeeze(0)
            aspect_probs = torch.sigmoid(aspect_logits).detach().cpu().numpy()
            selected_aspect_idxs = list((aspect_probs >= aspect_conf_thresh).nonzero()[0])

            if len(selected_aspect_idxs) == 0:
                selected_aspect_idxs = [int(torch.argmax(aspect_logits).item())]

            for a_idx in selected_aspect_idxs:
                asp_emb = model.aspect_emb(torch.tensor(a_idx, device=device)).unsqueeze(0)
                cat_input = torch.cat([fusion, asp_emb], dim=-1)
                cat_logits = model.cat_classifier(cat_input).squeeze(0)
                cat_probs = torch.sigmoid(cat_logits).detach().cpu().numpy()
                selected_cat_idxs = list((cat_probs >= cat_conf_thresh).nonzero()[0])

                if len(selected_cat_idxs) == 0:
                    selected_cat_idxs = [int(torch.argmax(cat_logits).item())]

                for c_idx in selected_cat_idxs:
                    cat_emb = model.cat_emb(torch.tensor(c_idx, device=device)).unsqueeze(0)
                    rel = torch.cat([span_emb, asp_emb, cat_emb], dim=-1)
                    rel = model.dropout(rel)
                    sent_logits = model.sent_classifier(rel)
                    sent_idx = int(torch.argmax(sent_logits, dim=-1).item())

                    evidences.append(
                        {
                            "char_span": (int(char_start), int(char_end)),
                            "aspect": bundle.aspects[a_idx],
                            "category": bundle.categories[c_idx],
                            "sentiment": bundle.sentiments[sent_idx],
                        }
                    )

        results.append({"evidences": evidences})

    return results


def infer_texts(
    texts: List[str],
    bundle: ModelBundle,
    batch_size: int = 8,
    max_len: int = 128,
    start_thresh: float = START_THRESH,
    end_thresh: float = END_THRESH,
    top_k_starts: int = TOP_K_STARTS,
    allow_overlap: bool = ALLOW_OVERLAP,
    temperature: float = SPAN_TEMPERATURE,
    length_penalty_coef: float = LENGTH_PENALTY_COEF,
    nms_iou_thresh: float = NMS_IOU_THRESH,
    max_candidates: int = MAX_CANDIDATES_PER_EXAMPLE,
    aspect_conf_thresh: float = ASPECT_CONF_THRESH,
    cat_conf_thresh: float = CAT_CONF_THRESH,
) -> List[Dict]:
    all_results: List[Dict] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = _prepare_batch(bundle.tokenizer, batch_texts, max_len=max_len)
        batch_results = _decode_batch(
            bundle,
            batch,
            start_thresh=start_thresh,
            end_thresh=end_thresh,
            top_k_starts=top_k_starts,
            allow_overlap=allow_overlap,
            temperature=temperature,
            length_penalty_coef=length_penalty_coef,
            nms_iou_thresh=nms_iou_thresh,
            max_candidates=max_candidates,
            aspect_conf_thresh=aspect_conf_thresh,
            cat_conf_thresh=cat_conf_thresh,
        )
        for text, res in zip(batch_texts, batch_results):
            evidences = []
            for ev in res["evidences"]:
                s, e = ev["char_span"]
                raw_span = text[s:e]
                span_text = raw_span.strip()
                if span_text == "":
                    span_text = raw_span
                ev_out = dict(ev)
                ev_out["span_text"] = span_text
                evidences.append(ev_out)
            all_results.append({"text": text, "evidences": evidences})
    return all_results


def hash_texts(texts: List[str]) -> str:
    joined = "\n".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()