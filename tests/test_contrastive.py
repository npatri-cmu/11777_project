import numpy as np
import pytest
from src.contrastive import combine, predict
from src.distortions import distort
from PIL import Image

def test_alpha_zero():
    x=[-10.,-2.,-3.]; y=[-1.,-100.,-3.]
    assert np.array_equal(combine(x,y,0),x)
    assert predict(['a','b','c'],combine(x,y,0))=='b'

def test_formula():
    assert np.allclose(combine([-2,-3],[-4,-2],.5),[-1,-3.5])

@pytest.mark.parametrize('x,y,a',[([1],[1,2],1),([float('nan')],[0],1),([0],[0],-1)])
def test_invalid(x,y,a):
    with pytest.raises(ValueError): combine(x,y,a)

def test_distortions():
    image=Image.new('RGB',(80,60),'red')
    for method in ['original','blur','grayscale']:
        out=distort(image,method)
        assert out.size==image.size and out.mode=='RGB'
    pixel=distort(image,'grayscale').getpixel((0,0))
    assert len(set(pixel))==1 and image.getpixel((0,0))==(255,0,0)
