import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = ImageOps.exif_transpose(img)  # fix phone image rotation
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)