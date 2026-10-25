import torch
import pytest
from src.scoring import answer_logprob

def test_teacher_forcing_multitoken():
    ids=torch.tensor([[0,1,2,3]])
    logits=torch.zeros(1,4,5)
    logits[0,1,2]=3.; logits[0,2,3]=2.
    expected=torch.log_softmax(logits[0,1],0)[2]+torch.log_softmax(logits[0,2],0)[3]
    assert torch.allclose(answer_logprob(logits,ids,2),expected.reshape(1))
    assert torch.allclose(answer_logprob(logits,ids,2,'mean'),expected.reshape(1)/2)
    # Prompt logits and last unused output must not affect scores.
    logits[0,0]=100.; logits[0,3]=100.
    assert torch.allclose(answer_logprob(logits,ids,2),expected.reshape(1))

def test_no_answer():
    with pytest.raises(ValueError): answer_logprob(torch.zeros(1,2,4),torch.zeros(1,2,dtype=torch.long),2)
