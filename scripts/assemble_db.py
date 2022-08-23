import json

def read_arr(filepath):
    with open(filepath, "r") as f:
        return json.loads(f.read())

def assemble_db(inputs_path, outputs_path, train_len, val_len, test_len):
    inputs = read_arr(inputs_path)
    outputs = read_arr(outputs_path)

    if len(inputs) != len(outputs):
        print("Error: inputs and outputs have different lengths")
        return
    if len(inputs) != train_len + val_len + test_len:
        print("Error: sub-datasets do not add up to total dataset")
        return

    dataset = {}
    dataset["inputs"] = inputs
    dataset["outputs"] = outputs
    dataset["train_len"] = train_len
    dataset["val_len"] = val_len
    dataset["test_len"] = test_len

    return dataset

def write_db(dataset, filepath):
    with open(filepath, "w") as f:
        json.dump(dataset, f)

def main():
    dataset = assemble_db("images3d.json", "labels3d.json", train_len=800, val_len=100, test_len=100)
    write_db(dataset, "dataset.json")

if __name__ == "__main__":
    main()