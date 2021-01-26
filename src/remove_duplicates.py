import hashlib
import os


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