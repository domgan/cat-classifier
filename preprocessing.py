import glob
import numpy as np
from PIL import Image
import os

size = 244


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

resize_aspect_fit("./Data/Cats", size)
resize_aspect_fit("Data/Not_cats", size)

cats = load_images("Data/Cats")
not_cats = load_images("Data/Not_Cats")
print(cats.shape, not_cats.shape)

# # chinatree_labels = labels(chinatree_images, 0)
# fig_labels = labels(fig_images, 0)
# judastree_labels = labels(judastree_images, 1)
# palm_labels = labels(palm_images, 2)
# pine_labels = labels(pine_images, 3)
#
#
# all_images = np.concatenate((fig_images, judastree_images, palm_images, pine_images), 0)
# all_labels = np.concatenate((fig_labels, judastree_labels, palm_labels, pine_labels), 0)
#
# shuffler = np.random.permutation(all_images.shape[0])
# all_images_shuffled = all_images[shuffler]
# all_labels_shuffled = labels_matrix(all_labels[shuffler], 4)
#
# split = 500
# train_data = all_images_shuffled[split:]
# train_labels = all_labels_shuffled[split:]
# test_data = all_images_shuffled[:split]
# test_labels = all_labels_shuffled[:split]
#
# print('Data loaded')
