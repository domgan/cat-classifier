import glob
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def resize(path, final_size):
    dirs = os.listdir(path)
    for item in dirs:
        file_path = os.path.join(path, item)
        if item == '.DS_Store': #or 'resized' in item:
            continue
        f, e = os.path.splitext(file_path)
        try:
            img = Image.open(file_path).convert('RGB')  # image extension *.png,*.jpg
        except UnidentifiedImageError:
            continue
        if img.size == (final_size, final_size):
            continue
        img = img.resize((final_size, final_size), Image.ANTIALIAS)
        img.save(f + 'resized.jpg', 'JPEG')  # format may what u want ,*.png,*jpg,*.gif
    return load_images(path, final_size)


def load_images(directory, size):
    result = []
    for f in glob.iglob(directory + "/*"):
        # if 'resized' not in f:
        #     continue
        try:
            img = np.asarray(Image.open(f), dtype=np.float32)[..., :3]
        except UnidentifiedImageError:
            continue
        if not img.shape == (size, size, 3):
            continue
        img /= 255.0
        result.append(img)
    result = np.stack(result, axis=0)
    return result


def labels(images, label):
    result = np.zeros(images.shape[0]) + label
    return result


def create_tensors(cats_path, not_cats_path, size=128):
    cats_images0 = resize("./Data/Cats/pack0", size)
    cats_images1 = resize("./Data/Cats/pack1", size)
    cats_images2 = resize("./Data/Cats/pack2", size)
    not_cats_images0 = resize("Data/Not_cats/pack0", size)
    not_cats_images1 = resize("Data/Not_cats/pack1", size)
    not_cats_images2 = resize("Data/Not_cats/pack2", size)
    not_cats_images_rand = resize("Data/Not_cats/pack_rand", size)


    cats_images = np.concatenate((cats_images0, cats_images1, cats_images2), 0)
    not_cats_images = np.concatenate((not_cats_images0, not_cats_images1, not_cats_images2, not_cats_images_rand), 0)
    print(str(cats_images.shape[0]), 'of cats')
    print(str(not_cats_images.shape[0]), 'of not cats')

    cats_labels = labels(cats_images, 1)
    not_cats_labels = labels(not_cats_images, 0)

    all_images = np.concatenate((cats_images, not_cats_images), 0)
    all_labels = np.concatenate((cats_labels, not_cats_labels), 0)
    # all_labels = to_categorical(all_labels, num_classes=2)

    train_data, test_data, train_labels, test_labels = \
        train_test_split(all_images, all_labels, test_size=0.10)

    # check: #
    from matplotlib import pyplot as plt
    rand = np.abs(np.random.randint(0, 200, 5))
    print(rand)
    for val in rand:
        plt.imshow(train_data[val])
        plt.title(str(train_labels[val]))
        plt.show()
    ##  ##  ##

    print(train_data.shape, train_labels.shape)
    print('Data loaded')
    return train_data, test_data, train_labels, test_labels


cats_path = "Data/training_set/cats"
not_cats_path = "Data/training_set/dogs"
train_data, test_data, train_labels, test_labels = \
    create_tensors(cats_path, not_cats_path)
