import glob
import numpy as np
from PIL import Image


def load_images(directory):
    result = []
    for f in glob.iglob(directory + "/*"):
        img = np.asarray(Image.open(f))
        if not img.shape == (224, 224, 3):
            continue
        img = img / 255
        result.append(img)
    result = np.stack(result, axis=0)
    return result


def labels(images, label):
    result = np.zeros(images.shape[0], dtype=int) + label
    return result


def labels_matrix(labels_list, labels_no):
    result = np.zeros((labels_no, len(labels_list)))
    for i in range(len(labels_list)):
        result[labels_list[i]][i] = 1
    return result.T


# chinatree_images = load_images("data_downloader/chinatree_cut")
fig_images = load_images("data_downloader/fig_cut")
judastree_images = load_images("data_downloader/judastree_cut")
palm_images = load_images("data_downloader/palm_cut")
pine_images = load_images("data_downloader/pine_cut")

# chinatree_labels = labels(chinatree_images, 0)
fig_labels = labels(fig_images, 0)
judastree_labels = labels(judastree_images, 1)
palm_labels = labels(palm_images, 2)
pine_labels = labels(pine_images, 3)


all_images = np.concatenate((fig_images, judastree_images, palm_images, pine_images), 0)
all_labels = np.concatenate((fig_labels, judastree_labels, palm_labels, pine_labels), 0)

shuffler = np.random.permutation(all_images.shape[0])
all_images_shuffled = all_images[shuffler]
all_labels_shuffled = labels_matrix(all_labels[shuffler], 4)

split = 500
train_data = all_images_shuffled[split:]
train_labels = all_labels_shuffled[split:]
test_data = all_images_shuffled[:split]
test_labels = all_labels_shuffled[:split]

print('Data loaded')