import json
from re import L
import numpy as np

def logit_convert(arr, max):
    if (len(arr.shape) != 1):
        print("Error: logit_convert only works on 1d arrays")
        return
    arr = np.expand_dims(arr, axis=len(arr.shape))
    arr = np.repeat(arr, 10, axis=1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (arr[i][j] == j):
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    return arr

def read_arr(filepath):
    with open(filepath, "r") as f:
        dict = json.loads(f.read())
        return np.array(dict)

def read_3d_arr(filepath, norm, logits):
    arr = np.round(read_arr(filepath) / norm, 3)
    if logits:
        arr = logit_convert(arr, 10)
    while(len(arr.shape) < 4):
        arr = np.expand_dims(arr, axis=1)
    return arr
    
def convert_json(src, dest, norm=1, logits=False):
    arr = read_3d_arr(src, norm, logits)
    with open(dest, 'w') as f:
        json.dump(arr.tolist(), f)

convert_json("images.json", "images3d.json", norm=255.0, logits=False)
convert_json("labels.json", "labels3d.json", norm=1.0, logits=True)