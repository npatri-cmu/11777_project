"""Report artifacts derived only from saved experimental evidence."""
from collections import Counter
import pandas as pd
from .utils import save, read

def export_report(cfg, manifest, rows, summary, out):
    splits={s:dict(pairs=len({r['pair_id'] for r in manifest['records'] if r['split']==s}),
                   images=sum(r['split']==s for r in manifest['records'])) for s in ['dev','eval']}
    method=dict(dataset=cfg['dataset_id'],revision=cfg['dataset_revision'],modalities=['RGB image','text question'],
        annotation='Reviewed acceptable color sets; grey canonicalized to gray',
        training='None: frozen pretrained model, no fine-tuning',validation=splits['dev'],test=splits['eval'],
        exclusions=manifest.get('excluded_pairs',cfg.get('excluded_pairs',[])),seed=cfg['seed'],
        model=cfg['model_id'],model_revision=cfg['model_revision'],candidate_vocabulary=manifest['candidates'],
        selection='Dev-only equal-weight ordinary/counterfactual accuracy; ties smaller alpha',
        enable_thinking=cfg['enable_thinking'],min_pixels=cfg['min_pixels'],
        scoring=cfg['scoring'],max_pixels=cfg['max_pixels'],blur_fraction=cfg['blur_fraction'])
    save(out/'methodology.json',method)
    root=out.parent
    run=read(root/'run.json')
    cache=list((root/'scores'/run['run_id']).glob('*.json'))
    timings=[read(p).get('seconds') for p in cache]
    known=[x for x in timings if x is not None]
    save(out/'compute_summary.json',dict(cached_conditions=len(cache),timed_conditions=len(known),
        completed_condition_seconds=sum(known) if known else None,
        hardware=read(root/'hardware.json') if (root/'hardware.json').exists() else None,
        caveat='Cache timings exclude loading, setup, idle time, failed attempts and repeated smoke runs; not scheduler allocation hours'))
    (out/'methodology.md').write_text(
        f"# Methodology facts for the main report\n\nInputs: RGB images and object-specific textual questions. "
        f"Dataset: {cfg['dataset_id']} (color), revision {cfg['dataset_revision']}. "
        f"This experiment uses {sum(v['pairs'] for v in splits.values())} paired examples. "
        f"Validation/dev: {splits['dev']['pairs']} pairs ({splits['dev']['images']} images); "
        f"held-out test/eval: {splits['eval']['pairs']} pairs ({splits['eval']['images']} images. "
        "No task training set or loss optimization is used: pretrained Qwen weights are frozen. "
        "The pretraining corpus is not our project training split. Pairs stay together; "
        "object categories may overlap across splits. Labels are reviewed acceptable-color sets.\n\n"
        "Baseline: teacher-forced candidate log-probability; contrastive score "
        "S(a)=(1+alpha)L(a|I,q)-alpha L(a|D(I),q). Select alpha using dev only. "
        "Blur-only and grayscale-only predictions are fixed input ablations, with no added GPU passes. "
        "This is candidate reranking, not full token-level VCD.\n\n"
        "Report counterfactual/ordinary accuracy, prior-consistent error rate, correction and regression "
        "with their denominators, and paired changes/intervals. Main-text discussion must interpret actual "
        "failure examples; detailed package versions may go in the appendix.\n")
    flat=[]
    for key,v in summary.items():
        entry={'method':key,**{k:x for k,x in v.items() if not isinstance(x,dict)}}
        for condition in ['all','ordinary','counterfactual']:
            ci=v.get('paired_change_'+condition)
            if ci:
                entry.update({f'{condition}_delta':ci['delta'],f'{condition}_ci_low':ci['ci95'][0],f'{condition}_ci_high':ci['ci95'][1]})
        flat.append(entry)
    df=pd.DataFrame(flat); df.to_csv(out/'metrics_table.csv',index=False)
    columns=['method','n','counterfactual_accuracy','ordinary_accuracy','prior_consistent_error_rate','correction_rate','regression_rate','all_delta','all_ci_low','all_ci_high']
    table=df.reindex(columns=columns)
    def value(x): return 'NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,float) else str(x)
    (out/'metrics_table.md').write_text('| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+
        '\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in table.itertuples(index=False,name=None))+'\n')
    table.to_latex(out/'metrics_table.tex',index=False,float_format='%.4f',escape=True)
    counts=Counter((r['method'],r['alpha'],r['image_condition'],r['category']) for r in rows)
    pd.DataFrame([dict(method=k[0],alpha=k[1],condition=k[2],category=k[3],count=v) for k,v in sorted(counts.items())]).to_csv(out/'failure_counts.csv',index=False)
    # Never overwrite manual annotations on analysis reruns.
    annotation=out/'error_review.csv'
    if not annotation.exists():
        pd.DataFrame([dict(example_id=r['example_id'],method=r['method'],alpha=r['alpha'],category=r['category'],
            image=r['image'],expected=';'.join(r['ground_truth']),prediction=r['prediction'],
            multimodal_concept='',observed_evidence='',hypothesized_cause='',alternative_explanation='',
            proposed_intervention='',followup_test='',reviewer='') for r in rows if r['method'] in ['blur','grayscale']]).to_csv(annotation,index=False)
    (out/'discussion_checklist.md').write_text(
        '# Results and discussion / updated ideas\n\n'
        '- Put the held-out metrics_table in the MAIN report, alongside accuracy.png.\n'
        '- Use failure_counts.csv and error_review.csv to select both successes and regressions.\n'
        '- Link each claim to example ID, image, full original/distorted candidate scores and alpha.\n'
        '- Distinguish visual ambiguity, object identification, chromatic evidence loss and typical-color errors. A typical-color match alone is not proof of causal prior reliance.\n'
        '- Discuss ordinary controls and blur-only/grayscale-only ablations before claiming better visual grounding.\n'
        '- Use dev alpha_sweep.png to explain parameter selection; do not retune on test outcomes.\n'
        '- For each updated idea: cite an observed failure, a multimodal concept, a mechanism, a mathematical hypothesis and a future evaluation that could falsify it.\n'
        '- Possible mechanisms, conditional on observed evidence: reliability-gated visual/text fusion; attribute-specific visual interventions; hidden-state grounding intervention. These are proposals, not implemented or validated improvements.\n'
        '- Reserve a new final evaluation set or clearly disclose repeated test use if test failures inspire later model changes.\n'
        '- This repository covers RQ2 only; the team still needs N-1 distinct research ideas.\n')
