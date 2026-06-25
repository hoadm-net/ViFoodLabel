#!/usr/bin/env python3
"""
De-tokenize BIO-tagged entities into phrase-level entities.

Converts token-level BIO sequences like:
  'NĂNG' (B-NUTRITION_NAME)
  'LƯỢNG/' (I-NUTRITION_NAME)
  'ENERGY' (I-NUTRITION_NAME)

Into phrase-level:
  'NĂNG LƯỢNG/ ENERGY' (NUTRITION_NAME)
"""

def detokenize_bio_entities(bio_entities: list) -> list:
    """
    Convert token-level BIO entities to phrase-level entities.

    Args:
        bio_entities: List of dicts with keys 'text', 'label'
                     where label is like 'B-NUTRITION_NAME' or 'I-NUTRITION_NAME'

    Returns:
        List of phrase-level entities with keys 'text', 'label'
    """

    if not bio_entities:
        return []

    phrases = []
    current_phrase = None

    for entity in bio_entities:
        text = entity['text']
        label = entity['label']

        # Parse BIO tag: "B-TYPE" or "I-TYPE"
        if label.startswith('B-'):
            # Begin tag - start new entity
            if current_phrase:
                # Save previous phrase
                phrases.append(current_phrase)

            entity_type = label[2:]  # Remove "B-" prefix
            current_phrase = {
                'text': text,
                'label': entity_type,
                'tokens': [text]
            }

        elif label.startswith('I-'):
            # Inside tag - continue current entity
            entity_type = label[2:]  # Remove "I-" prefix

            if current_phrase and current_phrase['label'] == entity_type:
                # Same type, append
                current_phrase['text'] += ' ' + text
                current_phrase['tokens'].append(text)
            else:
                # Type mismatch or no current phrase - start new one
                if current_phrase:
                    phrases.append(current_phrase)
                current_phrase = {
                    'text': text,
                    'label': entity_type,
                    'tokens': [text]
                }

        elif label == 'O':
            # No entity - save current if exists
            if current_phrase:
                phrases.append(current_phrase)
                current_phrase = None

    # Don't forget last phrase
    if current_phrase:
        phrases.append(current_phrase)

    # Clean up output - remove token tracking
    return [
        {
            'text': p['text'],
            'label': p['label']
        }
        for p in phrases
    ]


# Punctuation/brackets trimmed at WORD EDGES only (kept verbatim in stored GT).
# Internal punctuation is preserved so Vietnamese decimal commas ("6,3") and
# additive code groups ("330,334") keep their meaning.
_EDGE_PUNCT = ".,;:!?…·•()[]{}\"'`«»“”‘’-/\\"


def normalize_text(text: str) -> str:
    """Normalize text for comparison only: lowercase, collapse whitespace, and
    strip punctuation/brackets at the edges of each word. The stored ground
    truth and predictions keep their original punctuation untouched.
    """
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*/\s*', '/', text)        # tidy bilingual "a / b" -> "a/b"
    words = (w.strip(_EDGE_PUNCT) for w in text.split(' '))
    return ' '.join(w for w in words if w).strip()


def evaluate_detokenized(ground_truth_bio: list, predicted: list) -> dict:
    """
    Evaluate using de-tokenized entities (phrase-level comparison).

    Args:
        ground_truth_bio: List of token-level entities with BIO labels
        predicted: List of phrase-level entities

    Returns:
        Dict with metrics
    """

    # De-tokenize ground truth
    gt_phrases = detokenize_bio_entities(ground_truth_bio)

    # Create sets for comparison (normalized text + label)
    gt_set = {(normalize_text(e['text']), e['label']) for e in gt_phrases}
    pred_set = {(normalize_text(e['text']), e['label']) for e in predicted}

    # Calculate metrics
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    recall = tp / len(gt_set) if gt_set else 0
    precision = tp / len(pred_set) if pred_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'gt_count': len(gt_phrases),
        'pred_count': len(predicted),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'gt_phrases': gt_phrases,
        'pred_phrases': predicted,
    }


if __name__ == "__main__":
    # Test
    bio_entities = [
        {'text': 'NĂNG', 'label': 'B-NUTRITION_NAME'},
        {'text': 'LƯỢNG/', 'label': 'I-NUTRITION_NAME'},
        {'text': 'ENERGY', 'label': 'I-NUTRITION_NAME'},
        {'text': '36', 'label': 'B-NUTRITION_VALUE'},
        {'text': 'kcal', 'label': 'I-NUTRITION_VALUE'},
    ]

    result = detokenize_bio_entities(bio_entities)

    print("Input (token-level):")
    for e in bio_entities:
        print(f"  '{e['text']}' → {e['label']}")

    print("\nOutput (phrase-level):")
    for e in result:
        print(f"  '{e['text']}' → {e['label']}")
