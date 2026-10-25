"""Tiny RANDOM Qwen3.5: test real hybrid multimodal forward, never accuracy."""
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from src.scoring import answer_logprob
from src.vlm import Scorer

def tiny_model():
    config=Qwen3_5Config(
        text_config=dict(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=['linear_attention','full_attention'],
            rope_parameters={'rope_type':'default','mrope_section':[1,1,2],'partial_rotary_factor':1.0}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
            out_hidden_size=32,num_position_embeddings=16,deepstack_visual_indexes=[]),
        image_token_id=20,video_token_id=23,vision_start_token_id=21,vision_end_token_id=22)
    return Qwen3_5ForConditionalGeneration(config).eval()

def inputs(ids):
    ids=torch.tensor([ids])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        mm_token_type_ids=(ids==20).long(),pixel_values=torch.zeros(4,1536),image_grid_thw=torch.tensor([[1,2,2]]))

def test_tiny_qwen_loss_agrees_with_candidate_scoring():
    model=tiny_model(); batch=inputs([1,21,20,22,3,4,5])
    labels=batch['input_ids'].clone(); labels[:,:5]=-100
    with torch.inference_mode():
        out=model(**batch,labels=labels,use_cache=False)
    actual=answer_logprob(out.logits,batch['input_ids'],5,'mean')
    assert torch.allclose(actual,-out.loss.reshape(1),atol=1e-6)

@pytest.mark.parametrize('reduction',['sum','mean'])
def test_shared_prefix_and_multitoken_match_reference(reduction):
    model=tiny_model(); prompt=[1,21,20,22,3]
    vocabulary={'red':[4],'blue':[5],'light blue':[6,5]}
    scorer=Scorer.__new__(Scorer);scorer.model=model;scorer.cfg={'scoring':reduction}
    scorer.processor=SimpleNamespace(tokenizer=lambda text,**kwargs:{'input_ids':torch.tensor([vocabulary[text]])})
    scorer.inputs=lambda image,question:inputs(prompt)
    actual=scorer.score(None,'fixture question',list(vocabulary))
    expected=[]
    with torch.inference_mode():
        for ids in vocabulary.values():
            batch=inputs(prompt+ids);out=model(**batch,use_cache=False)
            expected.append(answer_logprob(out.logits,batch['input_ids'],len(prompt),reduction).item())
    assert torch.allclose(torch.tensor(actual),torch.tensor(expected),atol=1e-5)
