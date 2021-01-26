import requests
import json
import os
from tqdm import tqdm
from src.remove_duplicates import remove_duplicates


def download_cats_images(path, imgNum):
    path = os.path.normpath(path)
    print('Download started...')
    for i in tqdm(range(imgNum)):
        link = json.loads(requests.get('https://aws.random.cat/meow').text)['file']
        img_data = requests.get(link).content
        with open(os.path.join(path, str(i)+'.jpg'), 'wb') as handler:
            handler.write(img_data)
    print('Done')


imgNum = 250
path = 'Data/Cats/pack_rand'
download_cats_images(path, imgNum)
remove_duplicates(path)
