from PIL import ImageFilter, ImageOps

def distort(image, method, blur_fraction=0.04):
    image = image.convert('RGB')
    if method == 'original':
        return image.copy()
    if method == 'blur':
        if blur_fraction <= 0:
            raise ValueError('Blur strength must be positive')
        return image.filter(ImageFilter.GaussianBlur(min(image.size)*blur_fraction))
    if method == 'grayscale':
        return ImageOps.grayscale(image).convert('RGB')
    raise ValueError(method)
