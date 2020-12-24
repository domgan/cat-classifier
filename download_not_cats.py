# Imports
import requests

imgNum = 4000
imgDim = 128

print('Download started...')
for i in range(imgNum):
	img_data = requests.get('https://picsum.photos/' + str(imgDim)).content
	with open('Data/Not_cats/pack_rand/' + str(i) + '.jpg', 'wb') as handler:
	    handler.write(img_data)
print('Done')


import hashlib
import os
from time import sleep

def remove_duplicate(path):
	# cwd = os.getcwd()
	# os.chdir(os.path.join(cwd, path))
	unique = {}
	for file in os.scandir(path):
		with open(path + file.name, 'rb') as f:
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
		sleep(0.02)

path = r'Data/Not_cats/pack_rand/'
remove_duplicate(path)
