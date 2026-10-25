import argparse
import time
import torch
from pathlib import Path
from PIL import Image
from src.runner import validated, identity
from src.utils import config, save, digest, file_hash
from src.contrastive import combine, predict
from src.distortions import distort

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml')
    p.add_argument('--gpu',action='store_true'); a=p.parse_args()
    if not a.gpu:
        import subprocess,sys
        subprocess.run([sys.executable,'-m','pytest','-q'],check=True); return
    cfg=config(a.config); root,m=validated(cfg)
    from src.vlm import Scorer
    model=Scorer(cfg); report=[]; run_id=identity(cfg,m)
    torch.cuda.reset_peak_memory_stats()
    save(root/'run.json',dict(run_id=run_id,config=cfg,manifest_digest=digest(m),manifest_sha256=file_hash(root/'manifest.json')))
    # Only development pairs: qualitative review cannot leak evaluation answers.
    for r in [r for r in m['records'] if r['split']=='dev'][:10]:
        image=Image.open(root/r['image']).convert('RGB'); candidates=r['candidate_answers']
        scores={}; timing={}
        for k in ['original','blur','grayscale']:
            torch.cuda.synchronize(); start=time.monotonic()
            scores[k]=model.score(distort(image,k,cfg['blur_fraction']),r['question'],candidates)
            torch.cuda.synchronize(); timing[k]=time.monotonic()-start
            save(root/'scores'/run_id/f"{r['example_id']}-{k}.json",dict(
                run_id=run_id,example_id=r['example_id'],method=k,candidates=candidates,
                scores=scores[k],seconds=timing[k]))
        assert predict(candidates,scores['original'])==predict(candidates,combine(scores['original'],scores['blur'],0))
        prompts=[model.inputs(distort(image,k,cfg['blur_fraction']),r['question']) for k in scores]
        for other in prompts[1:]:
            assert prompts[0]['input_ids'].equal(other['input_ids'])
            assert prompts[0]['image_grid_thw'].equal(other['image_grid_thw'])
            assert prompts[0]['mm_token_type_ids'].equal(other['mm_token_type_ids'])
        report.append(dict(**r,scores=scores,seconds_per_condition=timing,generation=model.generate(image,r['question']),
            plausible_score_review_passed=False))
    save(root/'gpu_smoke.json',{'run_id':run_id,'examples':report,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})
    print('Review gpu_smoke.json against images: inspect token scores, plausible answers and generation. No automatic correctness claim.')
if __name__=='__main__': main()
