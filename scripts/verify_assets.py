"""Network-enabled real dataset/processor verification, without model weights."""
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig
from PIL import Image
from src.preprocessing import load_processor, prepare, messages
from src.data import labels, normalize
from src.distortions import distort
from src.utils import config, save

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); a=p.parse_args()
    cfg=config(a.config)
    ds=load_dataset(cfg['dataset_id'],revision=cfg['dataset_revision'],split='color')
    assert len(ds)==493
    vocabulary=sorted({x for row in ds for k in ['correct_answer','incorrect_answer'] for x in labels(row[k])})
    overlaps=[]
    for i,row in enumerate(ds):
        if set(labels(row['correct_answer'])) & set(labels(row['incorrect_answer'])):
            overlaps.append(i)
        else:
            normalize(row,i,'unassigned',vocabulary)
        for k in ['original_image','counterfact_image']:
            im=row[k].convert('RGB'); im.load(); assert min(im.size)>0
    assert overlaps == cfg['excluded_pairs'], f'Unexpected overlapping labels: {overlaps}'
    processor=load_processor(cfg)
    model_config=AutoConfig.from_pretrained(cfg['model_id'],revision=cfg['model_revision'])
    assert model_config.model_type=='qwen3_5'
    tensors=[]
    for method in ['original','blur','grayscale']:
        image=distort(ds[0]['original_image'],method,cfg['blur_fraction'])
        tensors.append(prepare(processor,image,'What color is the American chameleon in this image? Answer with only the color.'))
    assert all(tensors[0]['input_ids'].equal(t['input_ids']) for t in tensors)
    assert all(tensors[0]['image_grid_thw'].equal(t['image_grid_thw']) for t in tensors)
    assert all(tensors[0]['mm_token_type_ids'].equal(t['mm_token_type_ids']) for t in tensors)
    probe=prepare(processor,Image.new('RGB',(1024,1024),'red'),'What color?')
    grid=probe['image_grid_thw'][0].tolist()
    effective_pixels=grid[1]*grid[2]*processor.image_processor.patch_size**2
    assert cfg['min_pixels'] <= effective_pixels <= cfg['max_pixels']
    prefix=processor.tokenizer.decode(tensors[0]['input_ids'][0],skip_special_tokens=False)
    assert prefix.endswith('<think>\n\n</think>\n\n'), repr(prefix[-100:])
    assert '<think>\n\n</think>' in prefix
    token_ids={c:processor.tokenizer.encode(c,add_special_tokens=False) for c in vocabulary}
    assert all(token_ids.values())
    report=dict(dataset_revision=cfg['dataset_revision'],model_revision=cfg['model_revision'],
        pairs=len(ds),decoded_images=2*len(ds),overlapping_label_pair_ids=overlaps,eligible_pairs=len(ds)-len(overlaps),vocabulary=vocabulary,candidate_token_ids=token_ids,
        processor=type(processor).__name__,model_type=model_config.model_type,
        identical_prompt_ids_and_image_grid=True,model_weights_loaded=False,
        enable_thinking=False,assistant_prefix_tail=prefix[-100:],image_processor_size=processor.image_processor.size,
        probe_image_grid=grid,probe_effective_pixels=effective_pixels,
        identical_multimodal_token_types=True,all_candidates_single_token=all(len(v)==1 for v in token_ids.values()),
        first_ten=[{k:ds[i][k] for k in ['object','correct_answer','incorrect_answer']} for i in range(10)])
    save(Path(cfg['output'])/'asset_verification.json',report)
    print('Real dataset and processor checks passed; visual review and GPU model verification remain separate.')
if __name__=='__main__': main()
