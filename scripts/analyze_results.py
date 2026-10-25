import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from src.utils import config, read, save
from src.runner import validated, analysis_identity
from src.contrastive import combine, predict
from src.reporting import export_report
from src.evaluation import category, metrics, paired_interval

def collect(cfg, m, root, split, settings):
    run_id=analysis_identity(cfg,m,root); rows=[]
    for r in m['records']:
        if r['split']!=split: continue
        cache=root/'scores'/run_id
        def scores(method):
            d=read(cache/f"{r['example_id']}-{method}.json")
            if d['run_id']!=run_id or d['candidates']!=r['candidate_answers']:
                raise ValueError('Mismatched cache')
            return d['scores']
        original=scores('original'); baseline=predict(r['candidate_answers'],original)
        for method,alpha in settings:
            distorted=scores(method.removesuffix('_only')) if method!='original' else original
            score=np.asarray(distorted) if method.endswith('_only') else combine(original,distorted,alpha)
            prediction=predict(r['candidate_answers'],score)
            ok=prediction in r['ground_truth']; base_ok=baseline in r['ground_truth']
            rows.append(dict(**r,method=method,alpha=alpha,prediction=prediction,
                baseline_prediction=baseline,correct=ok,baseline_correct=base_ok,
                category=category(base_ok,ok),candidate_scores=score.tolist(),original_scores=original,distorted_scores=distorted,score_margin=float(np.sort(score)[-1]-np.sort(score)[-2])))
    if not rows: raise ValueError('No rows')
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml')
    p.add_argument('--phase',choices=['dev','eval'],required=True)
    a=p.parse_args(); cfg=config(a.config); root,m=validated(cfg); run_id=analysis_identity(cfg,m,root)
    if a.phase=='dev':
        settings=[('original',0.0),('blur_only',0.0),('grayscale_only',0.0)]+[(d,float(x)) for d in ['blur','grayscale'] for x in cfg['alphas']]
    else:
        selection=read(root/'selection.json')
        if selection['run_id']!=run_id: raise ValueError('Stale selection')
        settings=[('original',0.0),('blur_only',0.0),('grayscale_only',0.0)]+[(k,v) for k,v in selection['alpha_by_method'].items()]
    rows=collect(cfg,m,root,a.phase,settings); summary={}
    for method,alpha in settings:
        subset=[r for r in rows if r['method']==method and r['alpha']==alpha]
        result=metrics(subset)
        result['paired_change_all']=paired_interval(subset,cfg['seed'])
        for condition in ['ordinary','counterfactual']:
            result['metrics_'+condition]=metrics([r for r in subset if r['image_condition']==condition])
            result['paired_change_'+condition]=paired_interval([r for r in subset if r['image_condition']==condition],cfg['seed'])
        summary[f'{method}:{alpha}']=result
    out=root/a.phase; out.mkdir(exist_ok=True)
    save(out/'metrics.json',summary); save(out/'predictions.json',rows)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    if a.phase=='dev':
        best={}
        for method in ['blur','grayscale']:
            # Fixed objective: mean ordinary and counterfactual dev accuracy. Tie -> smaller alpha.
            best[method]=max(sorted(map(float,cfg['alphas'])),key=lambda x:
                (summary[f'{method}:{x}']['ordinary_accuracy']+summary[f'{method}:{x}']['counterfactual_accuracy'])/2)
        save(root/'selection.json',dict(run_id=run_id,alpha_by_method=best,
            objective='equal-weight ordinary/counterfactual dev accuracy; ties smallest alpha'))
    fig,ax=plt.subplots(figsize=(8,4))
    keys=list(summary); x=np.arange(len(keys)); width=.38
    for j,c in enumerate(['ordinary','counterfactual']):
        ax.bar(x+j*width,[summary[k][c+'_accuracy'] for k in keys],width,label=c)
    ax.set(ylim=(0,1),ylabel='Accuracy',title=f'{a.phase}: identical paired examples')
    ax.set_xticks(x+width/2,keys,rotation=55,ha='right'); ax.legend(); fig.tight_layout()
    fig.savefig(out/'accuracy.png',dpi=160); plt.close(fig)
    if a.phase=='dev':
        fig,ax=plt.subplots()
        for method in ['blur','grayscale']:
            for condition in ['ordinary','counterfactual']:
                xs=sorted(map(float,cfg['alphas']))
                ax.plot(xs,[summary[f'{method}:{v}'][condition+'_accuracy'] for v in xs],marker='o',label=f'{method} {condition}')
        ax.set(xlabel='Alpha',ylabel='Accuracy',ylim=(0,1)); ax.legend(); fig.tight_layout()
        fig.savefig(out/'alpha_sweep.png',dpi=160); plt.close(fig)
    representatives=[]
    for method in ['blur','grayscale']:
        for cat in ['corrected','regressed','both_correct','both_incorrect']:
            match=next((r for r in rows if r['method']==method and r['category']==cat),None)
            if match: representatives.append(match)
    if representatives:
        panel=Image.new('RGB',(800,260*len(representatives)),'white'); draw=ImageDraw.Draw(panel)
        for j,r in enumerate(representatives):
            im=Image.open(root/r['image']); im.thumbnail((300,240)); panel.paste(im,(0,260*j))
            text=f"{r['example_id']} {r['method']} alpha={r['alpha']}\n{r['category']}\nTruth: {r['ground_truth']}\nBase: {r['baseline_prediction']} New: {r['prediction']}"
            draw.multiline_text((310,260*j+10),text,fill='black')
        panel.save(out/'failure_grid.jpg')
    save(out/'representatives.json',representatives)
    export_report(cfg,m,rows,summary,out)
    print(f'Wrote {a.phase} metrics and plots; missing categories are not invented.')
if __name__=='__main__': main()
