from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


class Checker:
    def __init__(self, size=128):
        self.size = size

    def _resize_and_load(self, path):
        if os.path.isfile(path):
            img = Image.open(path)  # image extension *.png,*.jpg
            img = img.resize((self.size, self.size), Image.ANTIALIAS)

            img = np.asarray(img, dtype=np.float32)[..., :3]

            img = img / 255
            return np.expand_dims(img, 0)

    def check_file(self, file_path):
        model = load_model('model.h5', compile=False)
        image = self._resize_and_load(file_path)

        out = model.predict(image)[0][0]
        print(out)
        if out < 0.5:
            title = 'That\'s not cat ;_;'
        else:
            title = 'That\'s cat!'

        plt.imshow(image[0], interpolation='nearest')
        plt.title(title)
        plt.show()


checker = Checker()
checker.check_file('Data/Test/test_cat0.JPG')
checker.check_file('Data/Test/test_cat1.JPG')
checker.check_file('Data/Test/test_cat2.JPG')
checker.check_file('Data/Test/test_not_cat0.JPG')
checker.check_file('Data/Test/test_not_cat1.JPG')
checker.check_file('Data/Test/test_not_cat2.JPG')
