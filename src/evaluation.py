import numpy as np

def category(base_ok, new_ok):
    return {(False,True):'corrected',(True,False):'regressed',
            (True,True):'both_correct',(False,False):'both_incorrect'}[(base_ok,new_ok)]

def metrics(rows):
    def rate(values): return float(np.mean(values)) if values else None
    cf = [r for r in rows if r['image_condition']=='counterfactual']
    ordinary = [r for r in rows if r['image_condition']=='ordinary']
    wrong = [r for r in rows if not r['baseline_correct']]
    right = [r for r in rows if r['baseline_correct']]
    prior = sum(not r['correct'] and r['prediction'] in r['typical_labels'] for r in cf)
    return dict(n=len(rows), counterfactual_accuracy=rate([r['correct'] for r in cf]),
        ordinary_accuracy=rate([r['correct'] for r in ordinary]),
        prior_consistent_error_rate=prior/len(cf) if cf else None,
        correction_rate=rate([r['correct'] for r in wrong]),
        regression_rate=rate([not r['correct'] for r in right]),
        initially_wrong=len(wrong), initially_right=len(right),
        corrected=sum(r['correct'] for r in wrong), regressed=sum(not r['correct'] for r in right))

def paired_interval(rows, seed=42, repetitions=2000):
    # Bootstrap pairs, preserving ordinary/counterfactual dependence.
    ids = sorted(set(r['pair_id'] for r in rows))
    deltas = np.array([np.mean([int(r['correct'])-int(r['baseline_correct'])
                               for r in rows if r['pair_id']==i]) for i in ids])
    if len(ids)<2: return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(repetitions,len(ids)), replace=True).mean(axis=1)
    return {'delta':float(deltas.mean()), 'ci95':np.quantile(samples,[.025,.975]).tolist(), 'pairs':len(ids)}
