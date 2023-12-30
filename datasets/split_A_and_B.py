import os
from glob import glob
from tqdm import tqdm
import cv2

def split(stage='train'):
    os.makedirs(f"/home/cas/home_ez/Datasets/night2day/{stage}A", exist_ok=True)
    os.makedirs(f"/home/cas/home_ez/Datasets/night2day/{stage}B", exist_ok=True)
    for img_path in tqdm(glob(f"/home/cas/home_ez/Datasets/night2day/{stage}/*")):
        img = cv2.imread(img_path)
        img_A = img[:, :256]
        img_B = img[:, 256:]
        cv2.imwrite(f"/home/cas/home_ez/Datasets/night2day/{stage}A/{os.path.basename(img_path)}", img_A)
        cv2.imwrite(f"/home/cas/home_ez/Datasets/night2day/{stage}B/{os.path.basename(img_path)}", img_B)
        # print(img.shape, imgA.shape, imgB.shape)

split(stage='train')
split(stage='val')
split(stage='test')
