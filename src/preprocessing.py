"""Shared pinned Qwen3.5 image and non-thinking chat configuration."""
from transformers import AutoProcessor

def load_processor(cfg):
    if cfg.get('enable_thinking') is not False:
        raise ValueError('This direct-answer baseline requires enable_thinking: false')
    if not 0 < cfg['min_pixels'] <= cfg['max_pixels']:
        raise ValueError('Invalid image pixel bounds')
    processor=AutoProcessor.from_pretrained(cfg['model_id'],revision=cfg['model_revision'])
    # Transformers 5 uses size bounds (pixel areas), not a max_pixels attribute.
    processor.image_processor.size={'shortest_edge':cfg['min_pixels'],'longest_edge':cfg['max_pixels']}
    return processor

def messages(image, question):
    return [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':question}]}]

def prepare(processor, image, question):
    return processor.apply_chat_template(messages(image,question), tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
