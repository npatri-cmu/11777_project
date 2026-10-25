import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
from src.data import labels, normalize, split_pairs
from src.distortions import distort
from src.utils import config, save, file_hash

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--force', action='store_true', help='Overwrite prepared manifest/review deliberately')
    p.add_argument('--smoke', action='store_true', help='Prepare only 10 pairs; separate config/output recommended')
    a = p.parse_args(); cfg = config(a.config)
    root = Path(cfg['output'])
    if (root/'manifest.json').exists() and not a.force:
        raise FileExistsError('Prepared experiment exists; use another output or explicit --force')
    (root/'images').mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg['dataset_id'], revision=cfg['dataset_revision'], split='color')
    if len(ds) != 493:
        raise ValueError(f'Expected 493 color pairs, got {len(ds)}; inspect upstream')
    candidates = sorted({x for row in ds for key in ['correct_answer','incorrect_answer'] for x in labels(row[key])})
    eligible = [i for i in range(len(ds)) if i not in cfg['excluded_pairs']]
    assigned = split_pairs(len(eligible), cfg['seed'], cfg['dev_fraction'], 10 if a.smoke else cfg['max_pairs'])
    selected = {eligible[i]: split for i, split in assigned.items()}
    records = []; review = []
    for i, split in selected.items():
        rows = normalize(ds[i], i, split, candidates)
        for r, field in zip(rows, ['original_image','counterfact_image']):
            im = ds[i][field].convert('RGB'); im.save(root/r['image'])
            r['image_sha256'] = file_hash(root/r['image'])
        records.extend(rows)
        review.append({'pair_id': i, 'verified': False, 'notes': '',
                       'ordinary_labels': rows[0]['ground_truth'], 'counterfactual_labels': rows[1]['ground_truth']})
    save(root/'manifest.json', {'dataset_revision': cfg['dataset_revision'], 'candidates': candidates, 'excluded_pairs': cfg['excluded_pairs'], 'records': records})
    save(root/'label_review.json', review)
    # Every pair gets an inspectable panel, including >=10 examples and both distortions.
    (root/'review').mkdir(exist_ok=True)
    for i in selected:
        pair = [r for r in records if r['pair_id']==i]
        panel = Image.new('RGB',(900,520),'white'); draw = ImageDraw.Draw(panel)
        for j,r in enumerate(pair):
            im = Image.open(root/r['image'])
            for k,method in enumerate(['original','blur','grayscale']):
                thumb = distort(im, method, cfg['blur_fraction']); thumb.thumbnail((290,210))
                panel.paste(thumb,(k*300,j*260+45))
                draw.text((k*300+5,j*260+5),f"{i} {r['image_condition']} {method}",fill='black')
                draw.text((k*300+5,j*260+22),str(r['ground_truth']),fill='black')
        panel.save(root/'review'/f'{i:04d}.jpg')
    print(f'Prepared {len(selected)} pairs. Visually inspect review/*.jpg; mark each pair verified in label_review.json.')
if __name__ == '__main__': main()
