import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv

path = 'generated_images'
imgs_path = 'unnumpy_imgs'

i = 0
for img in tqdm(os.listdir(path)):
    img = (np.load(os.path.join(path, img)).reshape(64, 64, 3) * 255).astype(np.uint8)
    cv.imwrite(f'unnumpy_imgs/{i}.jpg', img)
    i+=1