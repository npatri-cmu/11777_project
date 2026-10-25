from types import SimpleNamespace
import pytest
from src.preprocessing import load_processor,prepare

def test_non_thinking_and_pixel_bounds(monkeypatch):
    seen={}
    class Processor:
        image_processor=SimpleNamespace(size={})
        def apply_chat_template(self,messages,**kwargs):
            seen.update(kwargs);return messages
    monkeypatch.setattr('src.preprocessing.AutoProcessor.from_pretrained',lambda *args,**kwargs:Processor())
    p=load_processor(dict(enable_thinking=False,min_pixels=65536,max_pixels=262144,model_id='fixture',model_revision='fixed'))
    assert p.image_processor.size=={'shortest_edge':65536,'longest_edge':262144}
    prepare(p,None,'What color?')
    assert seen['enable_thinking'] is False and seen['add_generation_prompt'] is True

def test_thinking_is_rejected():
    with pytest.raises(ValueError,match='enable_thinking'):
        load_processor({'enable_thinking':True})
