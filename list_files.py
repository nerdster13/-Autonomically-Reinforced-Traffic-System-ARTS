import os
from os import walk

def list_files():
    wd = os.getcwd() + "/imgs"

    files = []
    for (dirpath, dirnames, filenames) in walk(wd):
        files.extend(filenames)
        break

    list_file = open('%s/images_list.txt'%(os.getcwd()), 'a')
    for f in files:
        list_file.write('%s/%s\n'%(wd, f))

if __name__ == "__main__":
    list_files()
