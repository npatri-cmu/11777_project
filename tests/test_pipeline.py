"""Synthetic cache fixture ONLY: never interpreted as experimental results."""
import json
import subprocess
import sys
from pathlib import Path
import yaml
from PIL import Image
from src.utils import save, read, file_hash, digest
from src.runner import identity, validated
from src.data import normalize

def test_cached_pipeline(tmp_path):
    cfg=yaml.safe_load(Path('config.yaml').read_text()); cfg['output']=str(tmp_path)
    (tmp_path/'images').mkdir(); records=[]; reviews=[]
    for i in range(4):
        rows=normalize(dict(object='fixture',correct_answer='red',incorrect_answer='blue'),i,'dev' if i<2 else 'eval',['blue','red'])
        for row in rows:
            Image.new('RGB',(16,16),row['ground_truth'][0]).save(tmp_path/row['image'])
            row['image_sha256']=file_hash(tmp_path/row['image'])
        records+=rows
        reviews.append(dict(pair_id=i,verified=True,ordinary_labels=['red'],counterfactual_labels=['blue']))
    m=dict(dataset_revision=cfg['dataset_revision'],candidates=['blue','red'],records=records)
    save(tmp_path/'manifest.json',m); save(tmp_path/'label_review.json',reviews)
    assert validated(cfg)[1]==m
    run_id=identity(cfg,m)
    save(tmp_path/'run.json',dict(run_id=run_id,config=cfg,manifest_digest=digest(m)))
    for r in records:
        for method in ['original','blur','grayscale']:
            save(tmp_path/'scores'/run_id/f"{r['example_id']}-{method}.json",dict(run_id=run_id,candidates=m['candidates'],scores=[-1,-2]))
    config_path=tmp_path/'config.yaml'; config_path.write_text(yaml.safe_dump(cfg))
    for phase in ['dev','eval']:
        subprocess.run([sys.executable,'-m','scripts.analyze_results','--config',str(config_path),'--phase',phase],check=True)
        assert (tmp_path/phase/'accuracy.png').stat().st_size>0
        for artifact in ['methodology.json','metrics_table.csv','metrics_table.md','metrics_table.tex','failure_counts.csv','error_review.csv']:
            assert (tmp_path/phase/artifact).stat().st_size>0
        assert read(tmp_path/phase/'methodology.json')['training'].startswith('None')
        assert 'blur_only:0.0' in read(tmp_path/phase/'metrics.json')
        assert read(tmp_path/phase/'metrics.json')['original:0.0']['n']==4
    assert read(tmp_path/'selection.json')['alpha_by_method']=={'blur':0.0,'grayscale':0.0}
    subprocess.run([sys.executable,'-m','scripts.run_vcd','--config',str(config_path),'--split','dev','--smoke'],check=True)
    cfg['max_pixels']*=2
    assert identity(cfg,m)!=run_id

def test_unverified_blocked(tmp_path):
    import pytest
    cfg={'output':str(tmp_path),'dataset_revision':'x'}
    rows=normalize(dict(object='fixture',correct_answer='red',incorrect_answer='blue'),0,'dev',['blue','red'])
    save(tmp_path/'manifest.json',dict(dataset_revision='x',candidates=['blue','red'],records=rows))
    save(tmp_path/'label_review.json',[dict(pair_id=0,verified=False)])
    with pytest.raises(ValueError,match='review incomplete'): validated(cfg)


def test_worker_shards_preserve_pairs_and_cover_all_rows():
    import pytest
    from src.runner import shard_rows
    rows = [dict(pair_id=i, example_id=f'{i}-{condition}')
            for i in [9, 2, 17, 4, 30] for condition in ['ordinary', 'counterfactual']]
    parts = [shard_rows(rows, rank, 2) for rank in range(2)]
    ids = [{r['example_id'] for r in part} for part in parts]
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == {r['example_id'] for r in rows}
    assert not {r['pair_id'] for r in parts[0]} & {r['pair_id'] for r in parts[1]}
    assert shard_rows(rows, 0, 1) == rows
    for rank, size in [(2, 2), (-1, 2), (0, 0), (0, 3)]:
        with pytest.raises(ValueError):
            shard_rows(rows, rank, size)
