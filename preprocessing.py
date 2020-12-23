import glob
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
from sklearn.model_selection import train_test_split

size = 200


def resize_delete_load(path, final_size):
    dirs = os.listdir(path)
    for item in dirs[:500]:
        file_path = os.path.join(path, item)
        if item == '.DS_Store' or 'resized' in item:
            continue
        if os.path.isfile(file_path):
            try:
                im = Image.open(file_path)
            except UnidentifiedImageError:
                continue
            f, e = os.path.splitext(file_path)
            size = im.size
            if size[0] < final_size or size[1] < final_size:
                continue
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
            new_im.save(f + 'resized.jpg', 'JPEG', quality=90)
    return load_images(path)


def resize(path, size):
    dirs = os.listdir(path)
    for item in dirs[:500]:
        if 'resized' in item:
            continue
        if os.path.isfile(path+item):
            try:
                im = Image.open(path+item)
            except UnidentifiedImageError:
                continue
            f, e = os.path.splitext(path+item)
            imResize = im.resize((size, size), Image.ANTIALIAS).convert('RGB')
            imResize.save(f+'resized.jpg', 'JPEG', quality=80)
    return load_images(path)


def load_images(directory):
    ctr = 0
    result = []
    for f in glob.iglob(directory + "/*"):
        ctr += 1
        if ctr > 1500:
            continue
        if 'resized' not in f:
            continue
        img = np.asarray(Image.open(f))
        if not img.shape == (size, size, 3):
            continue
        img = img / 255
        result.append(img)
    result = np.stack(result, axis=0)
    return result


def labels(images, label):
    result = np.zeros(images.shape[0], dtype=int) + label
    return result


# cats_images0 = resize_delete_load("./Data/Cats/pack0", size)
# cats_images1 = resize_delete_load("./Data/Cats/pack1", size)
# not_cats_images0 = resize_delete_load("Data/Not_cats/pack0", size)
# not_cats_images1 = resize_delete_load("Data/Not_cats/pack1", size)
#
# cats_images = np.concatenate((cats_images0, cats_images1), 0)
# not_cats_images = np.concatenate((not_cats_images0, not_cats_images1), 0)

cats_images = resize("Data/PetImages/Cat/", size)
not_cats_images = resize("Data/PetImages/Dog/", size)

cats_labels = labels(cats_images, 1)
not_cats_labels = labels(not_cats_images, 0)

all_images = np.concatenate((cats_images, not_cats_images), 0)
all_labels = np.concatenate((cats_labels, not_cats_labels), 0)

train_data, test_data, train_labels, test_labels = \
    train_test_split(all_images, all_labels, test_size=0.20)


# check:
from matplotlib import pyplot as plt
plt.imshow(train_data[0], interpolation='nearest')
plt.show()
print(train_data.shape[0], train_labels[0])

print('Data loaded')
