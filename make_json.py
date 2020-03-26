import json
import os
import sys

def makeJSON(dir, full_path):
    print('Making JSON for', dir)
    dir_list = os.listdir(full_path)
    data = {}
    for d in dir_list:
        file_list = os.listdir(os.path.join(full_path, d))
        if 'new.stl' in file_list:
            data[d] = os.path.join(full_path, d+'/new.stl')
    with open(dir+".json", 'w') as f:
         json.dump(data, f)

if __name__ == "__main__":
    makeJSON(sys.argv[1], sys.argv[2])