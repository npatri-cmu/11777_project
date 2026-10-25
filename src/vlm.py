import torch
import os
from transformers import Qwen3_5ForConditionalGeneration
from .preprocessing import load_processor, prepare
from .scoring import answer_logprob

class Scorer:
    def __init__(self, cfg):
        if cfg['device'] != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Real inference requires a CUDA GPU; CPU tests use no weights')
        device_index = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(device_index)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unavailable; request a BF16-capable cluster GPU and compatible CUDA build')
        if cfg['batch_size'] != 1:
            raise ValueError('This memory-bounded implementation supports batch_size=1')
        if cfg['dtype'] != 'bfloat16':
            raise ValueError('Baseline is explicitly BF16')
        torch.manual_seed(cfg['seed'])
        self.cfg = cfg
        self.processor = load_processor(cfg)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg['model_id'], revision=cfg['model_revision'], dtype=torch.bfloat16,
            device_map={'': device_index}, attn_implementation='sdpa').eval()
        self.model.requires_grad_(False)

    def inputs(self, image, question):
        return prepare(self.processor,image,question).to(self.model.device)

    @torch.inference_mode()
    def score(self, image, question, candidates):
        base = self.inputs(image, question)
        p = base['input_ids'].shape[1]
        answer_ids=[self.processor.tokenizer(c,add_special_tokens=False,return_tensors='pt')['input_ids'].to(self.model.device) for c in candidates]
        if any(ids.shape[1]==0 for ids in answer_ids):
            raise ValueError('Empty candidate tokenization')
        # All one-token candidates share the exact same conditional distribution.
        # Compute it once; longer answers still use full teacher forcing below.
        first_logprobs=None
        if any(ids.shape[1]==1 for ids in answer_ids):
            first_logprobs=self.model(**base,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)
        scores=[]
        for ids in answer_ids:
            if ids.shape[1]==1:
                scores.append(first_logprobs[ids[0,0]].item())
                continue
            inputs=dict(base)
            inputs['input_ids']=torch.cat([base['input_ids'],ids],dim=1)
            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
            if 'mm_token_type_ids' in base:
                inputs['mm_token_type_ids']=torch.cat([base['mm_token_type_ids'],torch.zeros_like(ids)],dim=1)
            output=self.model(**inputs,use_cache=False)
            scores.append(answer_logprob(output.logits,inputs['input_ids'],p,self.cfg['scoring']).item())
            del output
        return scores

    @torch.inference_mode()
    def generate(self, image, question):
        inputs = self.inputs(image, question)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=16)
        return self.processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
