#!/usr/bin/env python3


import numpy as np
from collections import defaultdict

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    linear_sum_assignment = None
    _HAS_SCIPY = False

EPS = 1e-12


def _span_is_valid(span):
    return (
        isinstance(span, (tuple, list))
        and len(span) == 2
        and span[0] is not None
        and span[1] is not None
    )


def _normalize_char_span(span):
    if not _span_is_valid(span):
        return None
    try:
        s = int(span[0])
        e = int(span[1])
    except Exception:
        return None
    if e < s:
        return None
    return (s, e)


def _normalize_evidence(ev):
    if not isinstance(ev, dict):
        ev = {}
    span = _normalize_char_span(ev.get("char_span"))
    return {
        "char_span": span,
        "aspect": ev.get("aspect", None),
        "category": ev.get("category", None),
        "sentiment": ev.get("sentiment", None),
    }


def _normalize_example(example):
    if not isinstance(example, dict):
        return {"text": None, "evidences": []}
    evidences = example.get("evidences", [])
    if evidences is None:
        evidences = []
    return {
        "text": example.get("text", None),
        "evidences": [_normalize_evidence(ev) for ev in evidences],
    }


def span_iou(span_a, span_b):
    """IoU on half-open char spans [s, e)."""
    span_a = _normalize_char_span(span_a)
    span_b = _normalize_char_span(span_b)
    if span_a is None or span_b is None:
        return 0.0

    s1, e1 = span_a
    s2, e2 = span_b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def span_exact_match(span_a, span_b):
    span_a = _normalize_char_span(span_a)
    span_b = _normalize_char_span(span_b)
    return span_a is not None and span_b is not None and span_a == span_b


def _hungarian_match(scores):
    """
    scores: numpy array shape (n_rows, n_cols) with similarity scores (bigger better).
    Returns list of (row_idx, col_idx) pairs maximizing total score.
    """
    n_rows, n_cols = scores.shape
    n = max(n_rows, n_cols)

    cost = np.zeros((n, n), dtype=float)
    cost[:n_rows, :n_cols] = -scores  # linear_sum_assignment minimizes

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n_rows and c < n_cols:
            pairs.append((int(r), int(c)))
    return pairs


def _greedy_match(scores):
    """
    Greedy one-to-one matching fallback.
    Returns list of (row_idx, col_idx) pairs ordered by score descending.
    """
    if scores.size == 0:
        return []

    flat = []
    n_rows, n_cols = scores.shape
    for r in range(n_rows):
        for c in range(n_cols):
            flat.append((float(scores[r, c]), r, c))
    flat.sort(key=lambda x: x[0], reverse=True)

    used_rows = set()
    used_cols = set()
    pairs = []

    for score, r, c in flat:
        if score <= 0:
            break
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        pairs.append((int(r), int(c)))
    return pairs


def _match_spans(pred_spans, gold_spans, score_fn, threshold=0.5):
    """
    Generic one-to-one matching between predicted and gold spans.

    Returns:
        matches: list of (pred_idx, gold_idx, score)
        matched_pred_idx: set
        matched_gold_idx: set
        score_matrix: np.ndarray
    """
    if len(pred_spans) == 0 or len(gold_spans) == 0:
        return [], set(), set(), np.zeros((len(pred_spans), len(gold_spans)), dtype=float)

    scores = np.zeros((len(pred_spans), len(gold_spans)), dtype=float)
    for i, p in enumerate(pred_spans):
        for j, g in enumerate(gold_spans):
            scores[i, j] = score_fn(p.get("char_span"), g.get("char_span"))

    if _HAS_SCIPY:
        pairs = _hungarian_match(scores)
    else:
        pairs = _greedy_match(scores)

    matches = []
    matched_pred_idx = set()
    matched_gold_idx = set()

    for pi, gi in pairs:
        sc = float(scores[pi, gi])
        if sc >= threshold:
            matches.append((pi, gi, sc))
            matched_pred_idx.add(pi)
            matched_gold_idx.add(gi)

    return matches, matched_pred_idx, matched_gold_idx, scores


def compute_metrics_from_counts(counts_dict):
    """
    counts_dict: {class_label: {"tp":int,"fp":int,"fn":int}, ...}

    Returns:
        {
          "macro": {
              "prec":,
              "rec":,
              "f1":,
              "weighted_prec":,
              "weighted_rec":,
              "weighted_f1":,
              "per_class": {...}
          },
          "micro": {"prec":, "rec":, "f1":}
        }
    """
    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_support = 0
    weighted_prec_sum = 0.0
    weighted_rec_sum = 0.0
    weighted_f1_sum = 0.0

    for cls, ct in counts_dict.items():
        tp = int(ct.get("tp", 0))
        fp = int(ct.get("fp", 0))
        fn = int(ct.get("fn", 0))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        f1 = (2 * prec * rec) / (prec + rec + EPS) if (prec + rec) > 0 else 0.0
        support = tp + fn

        per_class_metrics[cls] = {
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "support": support,
        }

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        total_support += support
        weighted_prec_sum += prec * support
        weighted_rec_sum += rec * support
        weighted_f1_sum += f1 * support

    if len(precisions) == 0:
        return {
            "macro": {
                "prec": 0.0,
                "rec": 0.0,
                "f1": 0.0,
                "weighted_prec": 0.0,
                "weighted_rec": 0.0,
                "weighted_f1": 0.0,
                "per_class": per_class_metrics,
            },
            "micro": {"prec": 0.0, "rec": 0.0, "f1": 0.0},
        }

    micro_prec = total_tp / (total_tp + total_fp + EPS)
    micro_rec = total_tp / (total_tp + total_fn + EPS)
    micro_f1 = (
        (2 * micro_prec * micro_rec) / (micro_prec + micro_rec + EPS)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    if total_support > 0:
        weighted_prec = weighted_prec_sum / total_support
        weighted_rec = weighted_rec_sum / total_support
        weighted_f1 = weighted_f1_sum / total_support
    else:
        weighted_prec = 0.0
        weighted_rec = 0.0
        weighted_f1 = 0.0

    return {
        "macro": {
            "prec": sum(precisions) / len(precisions),
            "rec": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s),
            "weighted_prec": weighted_prec,
            "weighted_rec": weighted_rec,
            "weighted_f1": weighted_f1,
            "per_class": per_class_metrics,
        },
        "micro": {
            "prec": micro_prec,
            "rec": micro_rec,
            "f1": micro_f1,
        },
    }


def debug_example_matching(text, preds, golds, per_example_matches, span_metric_name="IoU"):
    """
    Prints one readable example.

    per_example_matches entries:
        pred_idx, gold_idx, score, aspect_pred, aspect_gold, cat_pred, cat_gold, sent_pred, sent_gold
    """
    print("\n" + "=" * 100)
    if text:
        print("TEXT:")
        print(text)
        print("-" * 100)

    print("🟢 GOLD EVIDENCES")
    if not golds:
        print("(none)")
    for i, g in enumerate(golds):
        s, e = g.get("char_span", (None, None))
        print(
            f"[G{i}] span={s}-{e} | aspect={g.get('aspect', None)} | "
            f"category={g.get('category', None)} | sentiment={g.get('sentiment', None)}"
        )

    print("\n🔵 PREDICTED EVIDENCES")
    if not preds:
        print("(none)")
    for i, p in enumerate(preds):
        s, e = p.get("char_span", (None, None))
        print(
            f"[P{i}] span={s}-{e} | aspect={p.get('aspect', None)} | "
            f"category={p.get('category', None)} | sentiment={p.get('sentiment', None)}"
        )

    matched_pred_idx = set(m["pred_idx"] for m in per_example_matches)
    matched_gold_idx = set(m["gold_idx"] for m in per_example_matches)

    print(f"\n🔗 MATCHES ({span_metric_name} and labels):")
    if len(per_example_matches) == 0:
        print("(no matches)")
    else:
        for m in per_example_matches:
            print(
                f"P{m['pred_idx']} ↔ G{m['gold_idx']} | {span_metric_name}={m['score']:.3f} | "
                f"aspect: {m.get('aspect_pred')} ↔ {m.get('aspect_gold')} | "
                f"cat: {m.get('cat_pred')} ↔ {m.get('cat_gold')} | "
                f"sent: {m.get('sent_pred')} ↔ {m.get('sent_gold')}"
            )

    print("\n❌ UNMATCHED PREDICTIONS:")
    any_unp = False
    for i in range(len(preds)):
        if i not in matched_pred_idx:
            any_unp = True
            p = preds[i]
            s, e = p.get("char_span", (None, None))
            print(
                f"P{i} span={s}-{e} | aspect={p.get('aspect', None)} | "
                f"cat={p.get('category', None)} | sent={p.get('sentiment', None)}"
            )
    if not any_unp:
        print("(none)")

    print("\n❌ UNMATCHED GOLDS:")
    any_ung = False
    for i in range(len(golds)):
        if i not in matched_gold_idx:
            any_ung = True
            g = golds[i]
            s, e = g.get("char_span", (None, None))
            print(
                f"G{i} span={s}-{e} | aspect={g.get('aspect', None)} | "
                f"cat={g.get('category', None)} | sent={g.get('sentiment', None)}"
            )
    if not any_ung:
        print("(none)")

    print("=" * 100 + "\n")


def evaluate_dataset(
    all_preds,
    all_golds,
    iou_threshold=0.5,
    verbose=True,
    show_examples=0,
    exact_match_threshold=1.0,
):
    """
    all_preds, all_golds: lists of per-example dicts:
      pred: {"text":..., "evidences":[{"char_span":(s,e),"aspect":str,"category":str,"sentiment":str}, ...]}
      gold: {"text":..., "evidences":[... same format ...]}

    Returns a metrics dict with:
      - span (IoU threshold)
      - span_exact
      - triplet / triplet_strict
      - aspect / category / sentiment / pair_cat_sent
      - error_breakdown
      - matched_examples

    Notes:
      - Triplet strict = span match AND aspect match AND category match AND sentiment match
      - If span matches but labels mismatch => counts as FP and FN
      - Aspect/category/sentiment y category+sentiment se evalúan condicionados a span matching
    """
    assert len(all_preds) == len(all_golds), "preds and golds must have same length"

    preds_norm = [_normalize_example(x) for x in all_preds]
    golds_norm = [_normalize_example(x) for x in all_golds]

    # IoU span counters
    total_pred_spans = 0
    total_gold_spans = 0
    total_TP_spans_iou = 0
    total_FP_spans_iou = 0
    total_FN_spans_iou = 0

    # Exact span counters
    total_TP_spans_exact = 0
    total_FP_spans_exact = 0
    total_FN_spans_exact = 0

    # Triplet strict counters
    triplet_tp = 0
    triplet_fp = 0
    triplet_fn = 0

    # Conditional correctness counters on matched spans
    aspect_correct = 0
    cat_correct = 0
    sent_correct = 0
    pair_cat_sent_correct = 0

    # per-class counts for macro metrics
    aspect_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    cat_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    sent_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    pair_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # error analysis
    error_breakdown = defaultdict(int)

    matched_examples = []
    debug_printed = 0

    def _register_unmatched_pred(p_e):
        nonlocal triplet_fp
        triplet_fp += 1
        aspect_counts[p_e.get("aspect")]["fp"] += 1
        cat_counts[p_e.get("category")]["fp"] += 1
        sent_counts[p_e.get("sentiment")]["fp"] += 1
        pair_counts[f"{p_e.get('category')}|||{p_e.get('sentiment')}"]["fp"] += 1
        error_breakdown["unmatched_prediction"] += 1

    def _register_unmatched_gold(g_e):
        nonlocal triplet_fn
        triplet_fn += 1
        aspect_counts[g_e.get("aspect")]["fn"] += 1
        cat_counts[g_e.get("category")]["fn"] += 1
        sent_counts[g_e.get("sentiment")]["fn"] += 1
        pair_counts[f"{g_e.get('category')}|||{g_e.get('sentiment')}"]["fn"] += 1
        error_breakdown["unmatched_gold"] += 1

    for ex_idx, (pred, gold) in enumerate(zip(preds_norm, golds_norm)):
        p_spans = pred.get("evidences", [])
        g_spans = gold.get("evidences", [])

        total_pred_spans += len(p_spans)
        total_gold_spans += len(g_spans)

        # --- IoU matching ---
        iou_matches, matched_pred_idx_iou, matched_gold_idx_iou, iou_scores = _match_spans(
            p_spans,
            g_spans,
            span_iou,
            threshold=iou_threshold,
        )

        TP_span_iou = len(iou_matches)
        FP_span_iou = len(p_spans) - len(matched_pred_idx_iou)
        FN_span_iou = len(g_spans) - len(matched_gold_idx_iou)

        total_TP_spans_iou += TP_span_iou
        total_FP_spans_iou += FP_span_iou
        total_FN_spans_iou += FN_span_iou

        # --- Exact matching ---
        exact_matches, matched_pred_idx_exact, matched_gold_idx_exact, _ = _match_spans(
            p_spans,
            g_spans,
            lambda a, b: 1.0 if span_exact_match(a, b) else 0.0,
            threshold=exact_match_threshold,
        )

        total_TP_spans_exact += len(exact_matches)
        total_FP_spans_exact += len(p_spans) - len(matched_pred_idx_exact)
        total_FN_spans_exact += len(g_spans) - len(matched_gold_idx_exact)

        # --- Triplet strict + label evaluation on IoU-matched spans ---
        per_example_matches = []
        classification_error = False

        for (pi, gi, sc) in iou_matches:
            p_e = p_spans[pi]
            g_e = g_spans[gi]

            asp_p = p_e.get("aspect", None)
            asp_g = g_e.get("aspect", None)
            cat_p = p_e.get("category", None)
            cat_g = g_e.get("category", None)
            sent_p = p_e.get("sentiment", None)
            sent_g = g_e.get("sentiment", None)

            a_match = asp_p == asp_g
            c_match = cat_p == cat_g
            s_match = sent_p == sent_g

            if a_match and c_match and s_match:
                triplet_tp += 1
                aspect_correct += 1
                cat_correct += 1
                sent_correct += 1
                pair_cat_sent_correct += 1

                aspect_counts[asp_g]["tp"] += 1
                cat_counts[cat_g]["tp"] += 1
                sent_counts[sent_g]["tp"] += 1
                pair_counts[f"{cat_g}|||{sent_g}"]["tp"] += 1

                error_breakdown["correct_all"] += 1
            else:
                triplet_fp += 1
                triplet_fn += 1
                classification_error = True
                error_breakdown["label_mismatch_any"] += 1

                # pattern breakdown
                wrong_parts = []
                if not a_match:
                    wrong_parts.append("aspect")
                if not c_match:
                    wrong_parts.append("category")
                if not s_match:
                    wrong_parts.append("sentiment")
                if wrong_parts:
                    error_breakdown["label_error:" + "+".join(wrong_parts)] += 1

                # aspect
                if a_match:
                    aspect_correct += 1
                    aspect_counts[asp_g]["tp"] += 1
                else:
                    if asp_p is not None:
                        aspect_counts[asp_p]["fp"] += 1
                    if asp_g is not None:
                        aspect_counts[asp_g]["fn"] += 1

                # category
                if c_match:
                    cat_correct += 1
                    cat_counts[cat_g]["tp"] += 1
                else:
                    if cat_p is not None:
                        cat_counts[cat_p]["fp"] += 1
                    if cat_g is not None:
                        cat_counts[cat_g]["fn"] += 1

                # sentiment
                if s_match:
                    sent_correct += 1
                    sent_counts[sent_g]["tp"] += 1
                else:
                    if sent_p is not None:
                        sent_counts[sent_p]["fp"] += 1
                    if sent_g is not None:
                        sent_counts[sent_g]["fn"] += 1

                # pair category+sentiment
                if c_match and s_match:
                    pair_cat_sent_correct += 1
                    pair_counts[f"{cat_g}|||{sent_g}"]["tp"] += 1
                else:
                    if cat_p is not None and sent_p is not None:
                        pair_counts[f"{cat_p}|||{sent_p}"]["fp"] += 1
                    if cat_g is not None and sent_g is not None:
                        pair_counts[f"{cat_g}|||{sent_g}"]["fn"] += 1

            per_example_matches.append(
                {
                    "pred_idx": pi,
                    "gold_idx": gi,
                    "score": sc,
                    "aspect_pred": asp_p,
                    "aspect_gold": asp_g,
                    "cat_pred": cat_p,
                    "cat_gold": cat_g,
                    "sent_pred": sent_p,
                    "sent_gold": sent_g,
                }
            )

        # unmatched predictions / golds
        for pi in range(len(p_spans)):
            if pi not in matched_pred_idx_iou:
                _register_unmatched_pred(p_spans[pi])

        for gi in range(len(g_spans)):
            if gi not in matched_gold_idx_iou:
                _register_unmatched_gold(g_spans[gi])

        matched_examples.append(
            {
                "pred_count": len(p_spans),
                "gold_count": len(g_spans),
                "iou_matches": per_example_matches,
                "TP_span_iou": TP_span_iou,
                "FP_span_iou": FP_span_iou,
                "FN_span_iou": FN_span_iou,
                "TP_span_exact": len(exact_matches),
                "FP_span_exact": len(p_spans) - len(matched_pred_idx_exact),
                "FN_span_exact": len(g_spans) - len(matched_gold_idx_exact),
                "TP_triplet_example": sum(
                    1
                    for m in per_example_matches
                    if (m["aspect_pred"] == m["aspect_gold"]
                        and m["cat_pred"] == m["cat_gold"]
                        and m["sent_pred"] == m["sent_gold"])
                ),
            }
        )

        if show_examples > 0 and debug_printed < show_examples:
            if FP_span_iou > 0 or FN_span_iou > 0 or classification_error:
                debug_example_matching(
                    text=gold.get("text", None),
                    preds=p_spans,
                    golds=g_spans,
                    per_example_matches=per_example_matches,
                    span_metric_name=f"IoU≥{iou_threshold:.2f}",
                )
                debug_printed += 1

    # --- aggregate span metrics ---
    span_iou_prec = total_TP_spans_iou / (total_TP_spans_iou + total_FP_spans_iou + EPS)
    span_iou_rec = total_TP_spans_iou / (total_TP_spans_iou + total_FN_spans_iou + EPS)
    span_iou_f1 = (
        2 * span_iou_prec * span_iou_rec / (span_iou_prec + span_iou_rec + EPS)
        if (span_iou_prec + span_iou_rec) > 0
        else 0.0
    )

    span_exact_prec = total_TP_spans_exact / (total_TP_spans_exact + total_FP_spans_exact + EPS)
    span_exact_rec = total_TP_spans_exact / (total_TP_spans_exact + total_FN_spans_exact + EPS)
    span_exact_f1 = (
        2 * span_exact_prec * span_exact_rec / (span_exact_prec + span_exact_rec + EPS)
        if (span_exact_prec + span_exact_rec) > 0
        else 0.0
    )

    # --- strict triplet metrics ---
    triplet_prec = triplet_tp / (triplet_tp + triplet_fp + EPS)
    triplet_rec = triplet_tp / (triplet_tp + triplet_fn + EPS)
    triplet_f1 = (
        2 * triplet_prec * triplet_rec / (triplet_prec + triplet_rec + EPS)
        if (triplet_prec + triplet_rec) > 0
        else 0.0
    )

    total_matched_iou = total_TP_spans_iou
    total_matched_iou_safe = total_matched_iou if total_matched_iou > 0 else 1

    # accuracies conditioned on span match (these are not global micro P/R/F1)
    aspect_acc_given_span = aspect_correct / total_matched_iou_safe if total_matched_iou > 0 else 0.0
    cat_acc_given_span = cat_correct / total_matched_iou_safe if total_matched_iou > 0 else 0.0
    sent_acc_given_span = sent_correct / total_matched_iou_safe if total_matched_iou > 0 else 0.0
    pair_cat_sent_acc_given_span = pair_cat_sent_correct / total_matched_iou_safe if total_matched_iou > 0 else 0.0

    aspect_metrics = compute_metrics_from_counts(aspect_counts)
    cat_metrics = compute_metrics_from_counts(cat_counts)
    sent_metrics = compute_metrics_from_counts(sent_counts)
    pair_metrics = compute_metrics_from_counts(pair_counts)

    results = {
        # Backward-compatible key for span-level IoU evaluation
        "span": {
            "predicted": total_pred_spans,
            "gold": total_gold_spans,
            "tp": total_TP_spans_iou,
            "fp": total_FP_spans_iou,
            "fn": total_FN_spans_iou,
            "precision_micro": span_iou_prec,
            "recall_micro": span_iou_rec,
            "f1_micro": span_iou_f1,
        },
        # Explicit alias
        "span_iou": {
            "predicted": total_pred_spans,
            "gold": total_gold_spans,
            "tp": total_TP_spans_iou,
            "fp": total_FP_spans_iou,
            "fn": total_FN_spans_iou,
            "precision": span_iou_prec,
            "recall": span_iou_rec,
            "f1": span_iou_f1,
            "threshold": iou_threshold,
        },
        "span_exact": {
            "predicted": total_pred_spans,
            "gold": total_gold_spans,
            "tp": total_TP_spans_exact,
            "fp": total_FP_spans_exact,
            "fn": total_FN_spans_exact,
            "precision": span_exact_prec,
            "recall": span_exact_rec,
            "f1": span_exact_f1,
            "threshold": exact_match_threshold,
        },
        # Backward-compatible strict triplet key
        "triplet": {
            "tp": triplet_tp,
            "fp": triplet_fp,
            "fn": triplet_fn,
            "precision": triplet_prec,
            "recall": triplet_rec,
            "f1": triplet_f1,
        },
        "triplet_strict": {
            "tp": triplet_tp,
            "fp": triplet_fp,
            "fn": triplet_fn,
            "precision": triplet_prec,
            "recall": triplet_rec,
            "f1": triplet_f1,
        },
        "aspect": {
            "accuracy_given_span": aspect_acc_given_span,
            "micro": aspect_metrics["micro"],
            "macro": aspect_metrics["macro"],
        },
        "category": {
            "accuracy_given_span": cat_acc_given_span,
            "micro": cat_metrics["micro"],
            "macro": cat_metrics["macro"],
        },
        "sentiment": {
            "accuracy_given_span": sent_acc_given_span,
            "micro": sent_metrics["micro"],
            "macro": sent_metrics["macro"],
        },
        "pair_cat_sent": {
            "accuracy_given_span": pair_cat_sent_acc_given_span,
            "micro": pair_metrics["micro"],
            "macro": pair_metrics["macro"],
        },
        "error_breakdown": dict(sorted(error_breakdown.items(), key=lambda kv: kv[0])),
        "matched_examples": matched_examples,
    }

    if verbose:
        print("\n=== EVALUATION ===")
        print(f"IoU threshold: {iou_threshold:.2f}")
        print(f"Exact span threshold: {exact_match_threshold:.2f}")

        print(
            f"\nSpan IoU-level: Predicted={total_pred_spans}, Gold={total_gold_spans}, TP={total_TP_spans_iou}, "
            f"FP={total_FP_spans_iou}, FN={total_FN_spans_iou}"
        )
        print(
            f"Span IoU micro P/R/F1: {span_iou_prec:.4f} / {span_iou_rec:.4f} / {span_iou_f1:.4f}"
        )

        print(
            f"\nSpan exact: Predicted={total_pred_spans}, Gold={total_gold_spans}, TP={total_TP_spans_exact}, "
            f"FP={total_FP_spans_exact}, FN={total_FN_spans_exact}"
        )
        print(
            f"Span exact P/R/F1: {span_exact_prec:.4f} / {span_exact_rec:.4f} / {span_exact_f1:.4f}"
        )

        print(
            f"\nTriplet strict (span + aspect + category + sentiment): "
            f"P/R/F1 = {triplet_prec:.4f} / {triplet_rec:.4f} / {triplet_f1:.4f}"
        )

        print(
            f"\nAccuracy conditioned on matched span:"
            f"\n  Aspect     = {aspect_acc_given_span:.4f}"
            f"\n  Category   = {cat_acc_given_span:.4f}"
            f"\n  Sentiment  = {sent_acc_given_span:.4f}"
            f"\n  Cat+Sent   = {pair_cat_sent_acc_given_span:.4f}"
        )

        def _print_table(title, metrics_dict):
            macro_dict = metrics_dict["macro"]
            micro_dict = metrics_dict["micro"]
            per = macro_dict.get("per_class", {})
            if not per:
                print(f"\n{title}: (no classes)")
                return

            items = sorted(per.items(), key=lambda kv: kv[1].get("support", 0), reverse=True)
            print(f"\n{title}")
            print("-" * 78)
            print(f"{'class':<36}{'support':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
            print("-" * 78)
            for cls, stats in items:
                sup = stats.get("support", 0)
                p = stats.get("prec", 0.0)
                r = stats.get("rec", 0.0)
                f = stats.get("f1", 0.0)
                print(f"{str(cls):<36}{sup:8d}{p:8.3f}{r:8.3f}{f:8.3f}")
            print("-" * 78)
            print(f"micro    P={micro_dict['prec']:.4f}  R={micro_dict['rec']:.4f}  F1={micro_dict['f1']:.4f}")
            print(
                f"macro    P={macro_dict['prec']:.4f}  R={macro_dict['rec']:.4f}  F1={macro_dict['f1']:.4f}"
            )
            print(
                f"weighted P={macro_dict['weighted_prec']:.4f}  "
                f"R={macro_dict['weighted_rec']:.4f}  F1={macro_dict['weighted_f1']:.4f}"
            )

        _print_table("ASPECT - per-class (support descending)", aspect_metrics)
        _print_table("CATEGORY - per-class (support descending)", cat_metrics)
        _print_table("SENTIMENT - per-class (support descending)", sent_metrics)

        pair_per = pair_metrics["macro"].get("per_class", {})
        if pair_per:
            items = sorted(pair_per.items(), key=lambda kv: kv[1].get("support", 0), reverse=True)
            print("\nCATEGORY+SENTIMENT - per-class (support descending)")
            print("-" * 96)
            print(f"{'category|sentiment':<44}{'support':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
            print("-" * 96)
            for key, stats in items:
                support = stats.get("support", 0)
                p = stats.get("prec", 0.0)
                r = stats.get("rec", 0.0)
                f = stats.get("f1", 0.0)
                print(f"{str(key):<44}{support:8d}{p:8.3f}{r:8.3f}{f:8.3f}")
            print("-" * 96)
            print(
                f"micro    P={pair_metrics['micro']['prec']:.4f}  "
                f"R={pair_metrics['micro']['rec']:.4f}  F1={pair_metrics['micro']['f1']:.4f}"
            )
            print(
                f"macro    P={pair_metrics['macro']['prec']:.4f}  "
                f"R={pair_metrics['macro']['rec']:.4f}  F1={pair_metrics['macro']['f1']:.4f}"
            )
            print(
                f"weighted P={pair_metrics['macro']['weighted_prec']:.4f}  "
                f"R={pair_metrics['macro']['weighted_rec']:.4f}  F1={pair_metrics['macro']['weighted_f1']:.4f}"
            )

        if error_breakdown:
            print("\nERROR BREAKDOWN")
            print("-" * 78)
            for k, v in sorted(error_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"{k:<40}{v:>8d}")
            print("-" * 78)

        print("\n(Per-class breakdowns are also available in results['...']['macro']['per_class'])")

    return results


# ---------------------------- Reporting helpers ----------------------------
def _fmt_mean_std(mean_value, std_value, decimals=4):
    return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"


def format_duration(seconds):
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    secs = total_seconds - (minutes * 60)
    return f"{minutes}m {secs:05.2f}s"


def _print_global_metric_block(title, mean_block, std_block):
    print(f"\n{title}")
    print("-" * 78)
    print(
        "Counts: "
        f"Predicted={_fmt_mean_std(mean_block['predicted'], std_block['predicted'], decimals=1)}, "
        f"Gold={_fmt_mean_std(mean_block['gold'], std_block['gold'], decimals=1)}, "
        f"TP={_fmt_mean_std(mean_block['tp'], std_block['tp'], decimals=1)}, "
        f"FP={_fmt_mean_std(mean_block['fp'], std_block['fp'], decimals=1)}, "
        f"FN={_fmt_mean_std(mean_block['fn'], std_block['fn'], decimals=1)}"
    )
    print(
        "P/R/F1: "
        f"{_fmt_mean_std(mean_block['precision'], std_block['precision'])} / "
        f"{_fmt_mean_std(mean_block['recall'], std_block['recall'])} / "
        f"{_fmt_mean_std(mean_block['f1'], std_block['f1'])}"
    )


def _print_triplet_metric_block(mean_block, std_block):
    print("\nEvaluacion estricta de la cuadrupla (IoU >= 0.50)")
    print("-" * 78)
    print(
        "Counts: "
        f"TP={_fmt_mean_std(mean_block['tp'], std_block['tp'], decimals=1)}, "
        f"FP={_fmt_mean_std(mean_block['fp'], std_block['fp'], decimals=1)}, "
        f"FN={_fmt_mean_std(mean_block['fn'], std_block['fn'], decimals=1)}"
    )
    print(
        "P/R/F1: "
        f"{_fmt_mean_std(mean_block['precision'], std_block['precision'])} / "
        f"{_fmt_mean_std(mean_block['recall'], std_block['recall'])} / "
        f"{_fmt_mean_std(mean_block['f1'], std_block['f1'])}"
    )


def _print_conditioned_accuracy_block(mean, std):
    print("\nExactitud condicionada al span")
    print("-" * 78)
    print(f"Aspect       = {_fmt_mean_std(mean['aspect']['accuracy_given_span'], std['aspect']['accuracy_given_span'])}")
    print(f"Category     = {_fmt_mean_std(mean['category']['accuracy_given_span'], std['category']['accuracy_given_span'])}")
    print(f"Sentiment    = {_fmt_mean_std(mean['sentiment']['accuracy_given_span'], std['sentiment']['accuracy_given_span'])}")
    print(f"Cat+Sent     = {_fmt_mean_std(mean['pair_cat_sent']['accuracy_given_span'], std['pair_cat_sent']['accuracy_given_span'])}")


def _print_task_metric_block(title, mean_block, std_block):
    print(f"\n{title}")
    print("-" * 78)
    print(
        "Accuracy conditioned on matched span: "
        f"{_fmt_mean_std(mean_block['accuracy_given_span'], std_block['accuracy_given_span'])}"
    )
    print(
        "micro    "
        f"P={_fmt_mean_std(mean_block['micro']['prec'], std_block['micro']['prec'])}  "
        f"R={_fmt_mean_std(mean_block['micro']['rec'], std_block['micro']['rec'])}  "
        f"F1={_fmt_mean_std(mean_block['micro']['f1'], std_block['micro']['f1'])}"
    )
    print(
        "macro    "
        f"P={_fmt_mean_std(mean_block['macro']['prec'], std_block['macro']['prec'])}  "
        f"R={_fmt_mean_std(mean_block['macro']['rec'], std_block['macro']['rec'])}  "
        f"F1={_fmt_mean_std(mean_block['macro']['f1'], std_block['macro']['f1'])}"
    )
    print(
        "weighted "
        f"P={_fmt_mean_std(mean_block['macro']['weighted_prec'], std_block['macro']['weighted_prec'])}  "
        f"R={_fmt_mean_std(mean_block['macro']['weighted_rec'], std_block['macro']['weighted_rec'])}  "
        f"F1={_fmt_mean_std(mean_block['macro']['weighted_f1'], std_block['macro']['weighted_f1'])}"
    )


def _print_per_class_block(title, mean_per_class, std_per_class):
    print(f"\n{title} - per-class (support descending)")
    print("-" * 122)
    print(f"{'class':<36}{'support':>20}{'prec':>22}{'rec':>22}{'f1':>22}")
    print("-" * 122)

    rows = []
    for cls, stats in mean_per_class.items():
        rows.append((stats.get("support", 0.0), cls))
    rows.sort(key=lambda item: (-item[0], item[1]))

    for _, cls in rows:
        mean_stats = mean_per_class[cls]
        std_stats = std_per_class.get(cls, {})
        print(
            f"{cls:<36}"
            f"{_fmt_mean_std(mean_stats.get('support', 0.0), std_stats.get('support', 0.0), decimals=1):>20}"
            f"{_fmt_mean_std(mean_stats.get('prec', 0.0), std_stats.get('prec', 0.0)):>22}"
            f"{_fmt_mean_std(mean_stats.get('rec', 0.0), std_stats.get('rec', 0.0)):>22}"
            f"{_fmt_mean_std(mean_stats.get('f1', 0.0), std_stats.get('f1', 0.0)):>22}"
        )
    print("-" * 122)


def print_fold_summary(fold_idx, total_folds, train_size, val_size, test_size, summary, train_time_seconds=None):
    print(f"\n=== FOLD {fold_idx} / {total_folds} ===")
    print(f"Sizes -> train: {train_size}, val: {val_size}, test: {test_size}")
    if train_time_seconds is not None:
        print(f"Training time: {format_duration(train_time_seconds)}")
    print(
        f"Span IoU P/R/F1: {summary['span_iou']['precision']:.4f} / "
        f"{summary['span_iou']['recall']:.4f} / {summary['span_iou']['f1']:.4f}"
    )
    print(
        f"Span exact P/R/F1: {summary['span_exact']['precision']:.4f} / "
        f"{summary['span_exact']['recall']:.4f} / {summary['span_exact']['f1']:.4f}"
    )
    print(
        f"Triplet strict P/R/F1: {summary['triplet_strict']['precision']:.4f} / "
        f"{summary['triplet_strict']['recall']:.4f} / {summary['triplet_strict']['f1']:.4f}"
    )
    print(
        "Accuracy conditioned on matched span: "
        f"Aspect={summary['aspect']['accuracy_given_span']:.4f} | "
        f"Category={summary['category']['accuracy_given_span']:.4f} | "
        f"Sentiment={summary['sentiment']['accuracy_given_span']:.4f} | "
        f"Cat+Sent={summary['pair_cat_sent']['accuracy_given_span']:.4f}"
    )
    print(
        f"Aspect micro/macro F1: {summary['aspect']['micro']['f1']:.4f} / "
        f"{summary['aspect']['macro']['f1']:.4f}"
    )
    print(
        f"Category micro/macro F1: {summary['category']['micro']['f1']:.4f} / "
        f"{summary['category']['macro']['f1']:.4f}"
    )
    print(
        f"Sentiment micro/macro F1: {summary['sentiment']['micro']['f1']:.4f} / "
        f"{summary['sentiment']['macro']['f1']:.4f}"
    )
    print(
        f"Category+Sentiment micro/macro F1: {summary['pair_cat_sent']['micro']['f1']:.4f} / "
        f"{summary['pair_cat_sent']['macro']['f1']:.4f}"
    )


def print_cv_summary(aggregate):
    mean = aggregate["mean"]
    std = aggregate["std"]

    print("\n=== 5-FOLD CROSS-VALIDATION SUMMARY ===")
    print("\nDesempeno de localizacion de evidencia")
    _print_global_metric_block("Span IoU", mean["span_iou"], std["span_iou"])
    _print_global_metric_block("Span exact", mean["span_exact"], std["span_exact"])
    _print_triplet_metric_block(mean["triplet_strict"], std["triplet_strict"])
    _print_conditioned_accuracy_block(mean, std)
    _print_task_metric_block("ASPECT", mean["aspect"], std["aspect"])
    _print_per_class_block("ASPECT", mean["aspect"]["macro"]["per_class"], std["aspect"]["macro"]["per_class"])
    _print_task_metric_block("CATEGORY", mean["category"], std["category"])
    _print_per_class_block("CATEGORY", mean["category"]["macro"]["per_class"], std["category"]["macro"]["per_class"])
    _print_task_metric_block("SENTIMENT", mean["sentiment"], std["sentiment"])
    _print_per_class_block("SENTIMENT", mean["sentiment"]["macro"]["per_class"], std["sentiment"]["macro"]["per_class"])
    _print_task_metric_block("CATEGORY+SENTIMENT", mean["pair_cat_sent"], std["pair_cat_sent"])
    _print_per_class_block(
        "CATEGORY+SENTIMENT",
        mean["pair_cat_sent"]["macro"]["per_class"],
        std["pair_cat_sent"]["macro"]["per_class"],
    )
