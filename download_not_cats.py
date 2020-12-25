# Imports
import requests
import hashlib
import os
from time import sleep

# change to os.path.join
def download_random_images(path, imgNum, imgDim):
	print('Download started...')
	for i in range(imgNum):
		img_data = requests.get('https://picsum.photos/' + str(imgDim)).content
		with open(path + '/' + str(i) + '.jpg', 'wb') as handler:
		    handler.write(img_data)
	print('Done')


def remove_duplicates(path):
	unique = {}
	for file in os.scandir(path):
		with open(path + '/' + file.name, 'rb') as f:
			filehash = hashlib.md5(f.read()).hexdigest()
			if filehash not in unique:
				unique[filehash] = file.name
			else:
				# Test print before removing
				print(f'Removing --> {unique[filehash]}')
				try:
					os.remove(path + unique[filehash])
				except FileNotFoundError:
					print('Already deleted')
					continue


imgNum = 4000
imgDim = 128
path = 'Data/Not_cats/pack_rand'
download_random_images(path, imgNum, imgDim)
remove_duplicates(path)
