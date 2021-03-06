import requests
import os
from tqdm import tqdm
from src.remove_duplicates import remove_duplicates


# change to os.path.join
def download_random_images(path, imgNum, imgDim):
    path = os.path.normpath(path)
    print('Download started...')
    for i in tqdm(range(imgNum)):
        img_data = requests.get('https://picsum.photos/' + str(imgDim)).content
        with open(os.path.join(path, str(i)+'.jpg'), 'wb') as handler:
            handler.write(img_data)
    print('Done')


imgNum = 4000
imgDim = 128
path = 'Data/Not_cats/pack_rand'
download_random_images(path, imgNum, imgDim)
remove_duplicates(path)
