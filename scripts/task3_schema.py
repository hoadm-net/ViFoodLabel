#!/usr/bin/env python3
"""
Task 3 (End-to-End KIE) schema, ground-truth builder, and field-level scorer.

Maps the BIO entity annotations + HAS_VALUE relations in a Label Studio item
to the structured JSON record defined in docs/benchmark-tasks.md, and scores a
predicted record against ground truth with field-level / set-based F1.

Schema (per product):
    product_name, net_weight, manufacturer, origin, mfg_date, expiry_date  -> str
    ingredients, additives, warnings                                       -> list[str]
    nutrition_value                                                        -> dict[str, str]
"""

from detokenize_bio import normalize_text

SINGLE_FIELDS = {
    'PRODUCT_NAME': 'product_name',
    'NET_WEIGHT': 'net_weight',
    'MANUFACTURER': 'manufacturer',
    'ORIGIN': 'origin',
    'MFG_DATE': 'mfg_date',
    'EXPIRY_DATE': 'expiry_date',
}
LIST_FIELDS = {
    'INGREDIENT': 'ingredients',
    'ADDITIVE': 'additives',
    'WARNING': 'warnings',
}
SINGLE_KEYS = list(SINGLE_FIELDS.values())
LIST_KEYS = list(LIST_FIELDS.values())
NUTRITION_KEY = 'nutrition_value'
ALL_KEYS = SINGLE_KEYS + LIST_KEYS + [NUTRITION_KEY]


def empty_record():
    rec = {k: '' for k in SINGLE_KEYS}
    rec.update({k: [] for k in LIST_KEYS})
    rec[NUTRITION_KEY] = {}
    return rec


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

def merge_entities(tokens):
    """Merge a reading-order token list into phrase entities.

    Returns a list of {'label', 'text', 'idx'} where idx are the token
    positions that compose the phrase. Used both to build the Task 3 record
    and to count entities per label for dataset metadata.
    """
    entities, cur = [], None
    for i, tok in enumerate(tokens):
        label = tok['label']
        text = tok['text']
        if label == 'O':
            if cur:
                entities.append(cur); cur = None
            continue
        etype = label[2:]
        if label[:1] == 'B' or cur is None or cur['label'] != etype:
            if cur:
                entities.append(cur)
            cur = {'label': etype, 'text': text, 'idx': [i]}
        else:
            cur['text'] = (cur['text'] + ' ' + text).strip()
            cur['idx'].append(i)
    if cur:
        entities.append(cur)
    return entities


def build_record_from_tokens(img):
    """Build a Task 3 structured record from a processed image record.

    `img` is one entry from data/processed/{train,val}.json: reading-order
    sorted `tokens` (each with text + BIO label) plus `relations` (HAS_VALUE
    pairs referencing token positions, e.g. "token_007"). B-/I- runs are
    merged into phrase entities and routed to schema fields; NUTRITION_NAME ->
    NUTRITION_VALUE relations populate nutrition_value.
    """
    entities = merge_entities(img['tokens'])

    tokid2ent = {}
    for ei, e in enumerate(entities):
        for ti in e['idx']:
            tokid2ent[f"token_{ti:03d}"] = ei

    rec = empty_record()
    for e in entities:
        if e['label'] in SINGLE_FIELDS:
            f = SINGLE_FIELDS[e['label']]
            rec[f] = (rec[f] + ' ' + e['text']).strip() if rec[f] else e['text']
        elif e['label'] in LIST_FIELDS:
            rec[LIST_FIELDS[e['label']]].append(e['text'])

    for rel in img.get('relations', []):
        ei, ej = tokid2ent.get(rel['from_id']), tokid2ent.get(rel['to_id'])
        if ei is None or ej is None:
            continue
        a, b = entities[ei], entities[ej]
        if a['label'] == 'NUTRITION_NAME' and b['label'] == 'NUTRITION_VALUE':
            rec[NUTRITION_KEY].setdefault(a['text'], b['text'])
        elif a['label'] == 'NUTRITION_VALUE' and b['label'] == 'NUTRITION_NAME':
            rec[NUTRITION_KEY].setdefault(b['text'], a['text'])

    return rec


# --------------------------------------------------------------------------- #
# Prediction coercion + scoring
# --------------------------------------------------------------------------- #

def coerce_record(pred):
    """Coerce an arbitrary model JSON dict into the schema shape."""
    rec = empty_record()
    if not isinstance(pred, dict):
        return rec
    for k in SINGLE_KEYS:
        v = pred.get(k, '')
        rec[k] = ' '.join(map(str, v)) if isinstance(v, list) else (str(v) if v else '')
    for k in LIST_KEYS:
        v = pred.get(k, [])
        if isinstance(v, str):
            v = [v]
        rec[k] = [str(x) for x in v if str(x).strip()] if isinstance(v, list) else []
    nv = pred.get(NUTRITION_KEY, {})
    if isinstance(nv, dict):
        rec[NUTRITION_KEY] = {str(kk): str(vv) for kk, vv in nv.items()}
    elif isinstance(nv, list):  # e.g. [{"name":..,"value":..}]
        out = {}
        for it in nv:
            if isinstance(it, dict):
                name = it.get('name') or it.get('label')
                val = it.get('value') or it.get('text')
                if name and val:
                    out[str(name)] = str(val)
        rec[NUTRITION_KEY] = out
    return rec


import re as _re

MATCH_THRESHOLD = 0.6  # token-overlap threshold for a text match


def token_overlap(a, b):
    """Jaccard-style token overlap on normalized text (both sides normalized).

    Tolerant of trailing punctuation, OCR typos in a few tokens, and dropped
    %DV suffixes — so a model that reads the label correctly is not penalized
    for cosmetic differences against the literal ground-truth transcription.
    """
    ta = set(_re.findall(r'\w+', normalize_text(str(a))))
    tb = set(_re.findall(r'\w+', normalize_text(str(b))))
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _match_counts(gt_items, pred_items, thr=MATCH_THRESHOLD):
    """Greedy bipartite matching: a pred matches its best unused gt if
    token_overlap >= thr. Returns (tp, fp, fn)."""
    gt_items = [x for x in gt_items if str(x).strip()]
    pred_items = [x for x in pred_items if str(x).strip()]
    matched = set()
    tp = 0
    for p in pred_items:
        best, best_i = 0.0, -1
        for i, g in enumerate(gt_items):
            if i in matched:
                continue
            s = token_overlap(g, p)
            if s > best:
                best, best_i = s, i
        if best >= thr and best_i >= 0:
            matched.add(best_i); tp += 1
    return tp, len(pred_items) - tp, len(gt_items) - tp


def score(gt, pred):
    """Field-level scoring with token-overlap matching (threshold 0.6).

    - single fields: 1 unit each (skipped if both empty); matched by token overlap
    - list fields:   greedy bipartite over normalized phrases
    - nutrition:     greedy bipartite over normalized "name: value" pairs
    """
    counts = {}  # field-group -> [tp, fp, fn]

    def add(group, tp, fp, fn):
        c = counts.setdefault(group, [0, 0, 0])
        c[0] += tp; c[1] += fp; c[2] += fn

    for k in SINGLE_KEYS:
        g, p = gt[k], pred[k]
        if not g and not p:
            continue
        tp, fp, fn = _match_counts([g] if g else [], [p] if p else [])
        add(k, tp, fp, fn)

    for k in LIST_KEYS:
        tp, fp, fn = _match_counts(gt[k], pred[k])
        add(k, tp, fp, fn)

    gt_nv = [f"{kk}: {vv}" for kk, vv in gt[NUTRITION_KEY].items()]
    pred_nv = [f"{kk}: {vv}" for kk, vv in pred[NUTRITION_KEY].items()]
    tp, fp, fn = _match_counts(gt_nv, pred_nv)
    add(NUTRITION_KEY, tp, fp, fn)

    ttp = sum(c[0] for c in counts.values())
    tfp = sum(c[1] for c in counts.values())
    tfn = sum(c[2] for c in counts.values())
    return {'fields': counts, 'tp': ttp, 'fp': tfp, 'fn': tfn, **prf(ttp, tfp, tfn)}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


PROMPT = """Extract key information from this Vietnamese food product label image.

Return ONLY a JSON object with exactly these keys:
{
  "product_name": "",
  "net_weight": "",
  "manufacturer": "",
  "origin": "",
  "mfg_date": "",
  "expiry_date": "",
  "ingredients": [],
  "additives": [],
  "warnings": [],
  "nutrition_value": {}
}

Rules:
- Single-value fields are strings; use "" if absent.
- ingredients, additives, warnings are lists of strings; use [] if absent.
- additives: include the full descriptive name together with its code, e.g.
  "Chất làm dày (1442)" or "Chất điều vị (621)" — not the code number alone.
- nutrition_value is an object mapping each nutrient name to its value. Keep the
  %DV (daily value) percentage if printed, e.g.
  {"Năng lượng": "550 kcal", "Carbohydrate": "20 g 7%", "Natri": "0 mg 0%"}.
- Transcribe text as printed, including any spelling on the label (keep Vietnamese
  diacritics). No extra keys, no commentary.
"""


if __name__ == "__main__":
    import json, random
    imgs = json.load(open('data/processed/train.json')) + json.load(open('data/processed/val.json'))
    random.seed(42)
    for img in random.sample(imgs, 2):
        print('=' * 60, img['id'])
        print(json.dumps(build_record_from_tokens(img), ensure_ascii=False, indent=1)[:800])
