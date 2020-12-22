import glob
import numpy as np
from PIL import Image
import os

size = 200


def resize_aspect_fit(path, final_size):
    dirs = os.listdir(path)
    for item in dirs:
        file_path = os.path.join(path, item)
        if item == '.DS_Store' or 'resized' in item:
            continue
        if os.path.isfile(file_path):
            im = Image.open(file_path)
            f, e = os.path.splitext(file_path)
            size = im.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
            new_im.save(f + 'resized.jpg', 'JPEG', quality=90)


def load_images(directory):
    result = []
    for f in glob.iglob(directory + "/*"):
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


# def labels_matrix(labels_list, labels_no):
#     result = np.zeros((labels_no, len(labels_list)))
#     for i in range(len(labels_list)):
#         result[labels_list[i]][i] = 1
#     return result.T

resize_aspect_fit("./Data/Cats/pack0", size)
resize_aspect_fit("./Data/Cats/pack1", size)
resize_aspect_fit("Data/Not_cats", size)

cats_images0 = load_images("Data/Cats/pack0")
cats_images1 = load_images("Data/Cats/pack1")
cats_images = np.concatenate((cats_images0, cats_images1), 0)
cats_labels = labels(cats_images, 1)

not_cats_images = load_images("Data/Not_cats")
not_cats_labels = labels(not_cats_images, 0)

all_images = np.concatenate((cats_images, not_cats_images), 0)
all_labels = np.concatenate((cats_labels, not_cats_labels), 0)

shuffler = np.random.permutation(all_images.shape[0])
all_images_shuffled = all_images[shuffler]
# all_labels_shuffled = labels_matrix(all_labels[shuffler], 2)
all_labels_shuffled = all_labels[shuffler]

# check:
from matplotlib import pyplot as plt
plt.imshow(all_images_shuffled[0], interpolation='nearest')
plt.show()
print(all_labels_shuffled[0])

split = 10
train_data = all_images_shuffled[split:]
train_labels = all_labels_shuffled[split:]
test_data = all_images_shuffled[:split]
test_labels = all_labels_shuffled[:split]

print('Data loaded')
