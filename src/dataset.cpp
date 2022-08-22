#include "dataset.hpp"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <fstream>
#include <array>
#include <iostream>

/*
 * Loads in a JSON dataset at the filepath specified with
 * following format.
 *
 * {
 *     inputs:[[[[],...],...],...],
 *     outputs:[[[[],...],...],...],
 *     train_len:x,
 *     val_len:y,
 *     test_len:z
 * }
 */
Dataset::Dataset(const std::string &filepath)
{
    std::cout << "Loading dataset from " << filepath << std::endl;
    std::ifstream infile(filepath);
    json js;
    infile >> js;

    std::cout << "Parsing dataset" << std::endl;

    inputs = js["inputs"];
    outputs = js["outputs"];
    int train_len = js["train_len"];
    int val_len = js["val_len"];
    int test_len = js["test_len"];
    splits = std::make_pair(train_len, train_len + val_len);

    std::cout << "Verifying dataset" << std::endl;

    if (inputs.size() != outputs.size())
    {
        std::cerr << "Inputs and outputs are not the same size" << std::endl;
    }
    if (train_len + val_len + test_len != inputs.size())
    {
        std::cerr << "Dataset length mismatch" << std::endl;
    }

    std::cout << "Cleanup dataset loading" << std::endl;
}

int Dataset::train_len()
{
    return splits.first;
}

int Dataset::val_len()
{
    return splits.second - splits.first;
}

int Dataset::test_len()
{
    return inputs.size() - splits.second;
}

std::pair<Tensor, Tensor> Dataset::train(int index)
{
    if (index >= splits.first)
    {
        std::cerr << "out of bounds access to training data" << std::endl;
    }
    return std::make_pair(inputs[index],outputs[index]);
}

std::pair<Tensor, Tensor> Dataset::val(int index)
{
    if (index >= splits.second - splits.first)
    {
        std::cerr << "out of bounds access to validation data" << std::endl;
    }
    return std::make_pair(inputs[index + splits.first], outputs[index + splits.first]);
}

std::pair<Tensor, Tensor> Dataset::test(int index)
{
    if (index >= inputs.size() - splits.second)
    {
        std::cerr << "out of bounds access to testing data" << std::endl;
    }
    return std::make_pair(inputs[index + splits.second], outputs[index + splits.second]);
}
