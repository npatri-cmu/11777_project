import pytest
from src.data import labels, normalize, split_pairs

def test_multi_label():
    row=dict(object='apple',correct_answer="['red', 'green']",incorrect_answer='black')
    a,b=normalize(row,1,'dev',['black','green','red'])
    assert a['ground_truth']==['green','red'] and b['ground_truth']==['black']
    assert a['question']==b['question']
    assert a['candidate_answers']==b['candidate_answers']
    assert 'black' not in a['question'] and 'red' not in a['question']

def test_split():
    s=split_pairs(493,42,.25,200)
    assert len(s)==200 and list(s.values()).count('dev')==50
    assert s==split_pairs(493,42,.25,200)
    assert set(i for i,x in s.items() if x=='dev').isdisjoint(i for i,x in s.items() if x=='eval')

@pytest.mark.parametrize('value',[[],"[]",None,"[1]",''])
def test_bad_labels(value):
    with pytest.raises((ValueError,TypeError)): labels(value)

def test_overlap():
    with pytest.raises(ValueError): normalize(dict(object='x',correct_answer='red',incorrect_answer='red'),0,'dev',['red'])

def test_spelling_alias():
    assert labels("['grey', 'gray']")==['gray']
