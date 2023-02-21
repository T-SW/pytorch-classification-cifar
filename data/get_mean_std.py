import numpy as np
import cv2
import os

filepath = r''
pathDir = os.listdir(filepath)

imgs = []
means, stdevs = [], []

for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename))
    print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.
    imgs.append(img)

for i in range(3):
    pixels = np.array(imgs)[:, :, i].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))


print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
