from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


def resize_and_load_file(path, final_size=200):
    if os.path.isfile(path):
        im = Image.open(path)
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x * ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))

        img = np.asarray(new_im)
        img = img / 255
        return np.expand_dims(img, 0)


def cat_check(file_path):
    model = load_model('model.h5')
    image = resize_and_load_file(file_path)

    out = model.predict(image)
    print(out)
    if out < 0.5:
        title = 'That\'s not cat ;_;'
    else:
        title = 'That\'s cat!'

    plt.imshow(image[0], interpolation='nearest')
    plt.title(title)
    plt.show()


cat_check('Data/Test/test_cat0.JPG')
cat_check('Data/Test/test_not_cat0.JPG')
cat_check('Data/Test/test_not_cat1.JPG')
