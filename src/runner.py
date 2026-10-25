import argparse
import importlib.metadata
import time
import os
from pathlib import Path
from PIL import Image
from .utils import config, read, save, digest, file_hash
from .distortions import distort

def identity(cfg, manifest):
    versions = {p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','Pillow','numpy','datasets','accelerate','tokenizers']}
    for package in ['flash-linear-attention','causal-conv1d']:
        try:
            versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package]=None
    source = {p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    return digest({'config':{k:v for k,v in cfg.items() if k != 'output'},'manifest':manifest,'versions':versions,'source':source})

def analysis_identity(cfg, manifest, root):
    """Read inference provenance so CPU analysis is portable across platforms."""
    run = read(root/'run.json')
    comparable = lambda c: {k:v for k,v in c.items() if k != 'output'}
    if comparable(run['config']) != comparable(cfg) or run['manifest_digest'] != digest(manifest):
        raise ValueError('Recorded run differs from config/manifest')
    return run['run_id']

def validated(cfg):
    root = Path(cfg['output']); m = read(root/'manifest.json')
    if m['dataset_revision'] != cfg['dataset_revision']:
        raise ValueError('Dataset revision mismatch; reprepare')
    if len({r['example_id'] for r in m['records']}) != len(m['records']):
        raise ValueError('Duplicate example IDs')
    pairs = {}
    for r in m['records']:
        pairs.setdefault(r['pair_id'], []).append(r)
        if not set(r['ground_truth']) <= set(m['candidates']) or not r['ground_truth']:
            raise ValueError('Ground truth outside vocabulary')
        if r['candidate_answers'] != m['candidates']:
            raise ValueError('Non-global candidates')
    for pair in pairs.values():
        if {r['image_condition'] for r in pair} != {'ordinary','counterfactual'} or len(pair) != 2:
            raise ValueError('Incomplete pair')
        if len({r['split'] for r in pair}) != 1 or len({r['question'] for r in pair}) != 1:
            raise ValueError('Pair split/prompt mismatch')
    review = {r['pair_id']:r for r in read(root/'label_review.json')}
    for r in m['records']:
        v = review[r['pair_id']]
        if v['verified'] is not True or v[r['image_condition']+'_labels'] != r['ground_truth']:
            raise ValueError('Visual label review incomplete or labels changed; update manifest and review consistently')
        if file_hash(root/r['image']) != r['image_sha256']:
            raise ValueError('Image hash mismatch')
    return root, m

def shard_rows(rows, rank, world_size):
    """Assign entire pairs to disjoint workers; GPU count does not change run identity."""
    if world_size not in (1, 2) or not 0 <= rank < world_size:
        raise ValueError('Use one or two workers with a valid rank')
    pairs = sorted({r['pair_id'] for r in rows})
    assigned = set(pairs[rank::world_size])
    return [r for r in rows if r['pair_id'] in assigned]


def main(methods):
    p = argparse.ArgumentParser()
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--split',choices=['dev','eval'],required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-seconds',type=float,default=3600)
    a = p.parse_args(); cfg = config(a.config); root,m = validated(cfg)
    if not a.smoke:
        smoke = read(root/'gpu_smoke.json')
        if smoke['run_id'] != identity(cfg,m) or len(smoke['examples']) < 5:
            raise ValueError('Run GPU smoke for this configuration before full inference')
        if not all(x['plausible_score_review_passed'] is True for x in smoke['examples']):
            raise ValueError('Manually review GPU smoke scores before full inference')
    if a.split=='eval' and not a.smoke:
        selection = read(root/'selection.json')
        if selection['run_id'] != identity(cfg,m):
            raise ValueError('Lock dev selection for this configuration first')
    run_id = identity(cfg,m); cache = root/'scores'/run_id
    cache.mkdir(parents=True,exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    shard_rows([], rank, world_size)
    if rank == 0:
        save(root/'run.json',{'run_id':run_id,'config':cfg,'manifest_sha256':file_hash(root/'manifest.json'),'manifest_digest':digest(m)})
    rows = [r for r in m['records'] if r['split']==a.split]
    if a.smoke: rows=rows[:10]
    rows = shard_rows(rows, rank, world_size)
    from .vlm import Scorer
    start=time.monotonic(); model=None
    for r in rows:
        image=Image.open(root/r['image']).convert('RGB')
        for method in methods:
            path=cache/f"{r['example_id']}-{method}.json"
            if path.exists():
                old=read(path)
                if old['candidates'] != r['candidate_answers'] or old['run_id'] != run_id:
                    raise ValueError('Invalid cache')
                continue
            if time.monotonic()-start >= a.max_seconds:
                print('Time guard reached; rerun to resume. Resubmit the scheduler job to finish remaining scores.'); return
            if model is None: model=Scorer(cfg)
            t=time.monotonic()
            scores=model.score(distort(image,method,cfg['blur_fraction']),r['question'],r['candidate_answers'])
            save(path,dict(run_id=run_id,example_id=r['example_id'],method=method,
                candidates=r['candidate_answers'],scores=scores,seconds=time.monotonic()-t))
            print(r['example_id'],method,round(time.monotonic()-t,2),flush=True)
