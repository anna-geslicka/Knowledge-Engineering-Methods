import os
import numpy as np
from PIL import Image
import pickle

label_count = 0
images = []
labels = []
for d in os.listdir('dataset'):
    subdir = os.path.join('dataset', d)
    print(d)
    for img in os.listdir(subdir):
        img_path = os.path.join(subdir, img)
        image = Image.open(img_path).convert('RGB')
        images.append(np.asarray(image))
        labels.append(label_count)
    label_count += 1

images = np.array(images) / 127.5 - 1.
labels = np.array(labels)
bin_dataset = (images, labels)

pickle.dump(bin_dataset, open("bin_dataset", "wb"))
