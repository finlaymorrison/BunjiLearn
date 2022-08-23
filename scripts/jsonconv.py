import operator
import functools
import json

def chunk(lst, n):
    return (lst[i:i+n] for i in range(0, len(lst), n))

def i_parse(bytes):
    return int.from_bytes(bytes, "big")

def arrayify(data, dims):
    if len(dims) == 1:
        return [i_parse(x) for x in chunk(data,1)]
    size = functools.reduce(operator.mul, dims[1:], 1)
    return [arrayify(x, dims[1:]) for x in chunk(data,size)]

def read_idx(path):
    image_data = []
    with open(path, "rb") as f:
        magic = i_parse(f.read(2))
        if (magic != 0):
            return
        type = i_parse(f.read(1))
        dim_count = i_parse(f.read(1))
        dims = [i_parse(f.read(4)) for i in range(dim_count)]
        bytes = f.read(functools.reduce(operator.mul, dims, 1))
        return arrayify(bytes, dims)

MAX_IMAGES = 1000

def main():
    images = read_idx("train-images.idx3-ubyte")
    with open('images.json', 'w') as f:
        json.dump(images[:MAX_IMAGES], f)
    
    labels = read_idx("train-labels.idx1-ubyte")
    with open('labels.json', 'w') as f:
        json.dump(labels[:MAX_IMAGES], f)

if __name__ == "__main__":
    main()
    