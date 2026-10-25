import ast
import random
import math

def labels(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            value = ast.literal_eval(value)
        else:
            value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f'Invalid labels: {value!r}')
    if any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError('Labels must be nonempty strings')
    return sorted(set('gray' if x.strip().lower() == 'grey' else x.strip().lower() for x in value))

def split_pairs(n, seed, dev_fraction, max_pairs=None):
    if n < 2 or not 0 < dev_fraction < 1:
        raise ValueError('Need >=2 pairs and 0 < dev_fraction < 1')
    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    ids = ids[:max_pairs] if max_pairs else ids
    if len(ids) < 2:
        raise ValueError('Need >=2 selected pairs')
    k = min(len(ids)-1, max(1, math.ceil(len(ids)*dev_fraction)))
    return {i: 'dev' if j < k else 'eval' for j, i in enumerate(ids)}

def normalize(row, i, split, candidates):
    typical, counter = labels(row['correct_answer']), labels(row['incorrect_answer'])
    if set(typical) & set(counter):
        raise ValueError(f'Overlapping typical/counterfactual labels at {i}')
    obj = row['object'].strip()
    if not obj:
        raise ValueError('Empty object')
    return [dict(example_id=f'{i:04d}-{condition}', pair_id=i, object=obj,
                 image_condition=condition, split=split,
                 image=f'images/{i:04d}-{condition}.png',
                 question=f'What color is the {obj} in this image? Answer with only the color.',
                 ground_truth=truth, typical_labels=typical, candidate_answers=candidates)
            for condition, truth in [('ordinary', typical), ('counterfactual', counter)]]
