#include "dataset.hpp"
#include "log.hpp"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <fstream>
#include <array>
#include <iostream>

namespace bunji
{

Tensor<double, 3> vec_conv(const std::vector<std::vector<std::vector<double>>> &vec)
{
    std::size_t d0 = vec.size();
    std::size_t d1 = vec[0].size();
    std::size_t d2 = vec[0][0].size();

    Tensor<double, 3> output({d0, d1, d2});

    for (int i = 0; i < d0; ++i)
    {
        for (int j = 0; j < d1; ++j)
        {
            for (int k = 0; k < d2; ++k)
            {
                output[i][j][k] = vec[i][j][k];
            }
        }
    }

    return output;
}

std::vector<Tensor<double, 3>> vecs_conv(const std::vector<std::vector<std::vector<std::vector<double>>>> &vecs)
{
    std::vector<Tensor<double, 3>> tensors;

    for (int i = 0; i < vecs.size(); ++i)
    {
        tensors.push_back(vec_conv(vecs[i]));
    }

    return tensors;
}

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
    BUNJI_DBG("loading dataset from {}", filepath);
    
    std::ifstream infile(filepath);
    json js;
    infile >> js;

    BUNJI_DBG("parsing dataset");

    std::vector<std::vector<std::vector<std::vector<double>>>> inputs_vec = js["inputs"];
    std::vector<std::vector<std::vector<std::vector<double>>>> outputs_vec = js["outputs"];

    inputs = vecs_conv(inputs_vec);
    outputs = vecs_conv(outputs_vec);

    int train_len = js["train_len"];
    int val_len = js["val_len"];
    int test_len = js["test_len"];
    splits = std::make_pair(train_len, train_len + val_len);

    BUNJI_DBG("verifying dataset");

    if (inputs.size() != outputs.size())
    {
        BUNJI_WRN("inputs and outputs are not the same size");
    }
    if (train_len + val_len + test_len != inputs.size())
    {
        BUNJI_WRN("incorrect amount of data for all dataset partitions");
    }

    BUNJI_DBG("cleanup dataset loading");
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

std::pair<Tensor<double, 3>, Tensor<double, 3>> Dataset::train(int index)
{
    if (index >= splits.first)
    {
        std::cerr << "out of bounds access to training data" << std::endl;
    }
    return std::make_pair(inputs[index],outputs[index]);
}

std::pair<Tensor<double, 3>, Tensor<double, 3>> Dataset::val(int index)
{
    if (index >= splits.second - splits.first)
    {
        std::cerr << "out of bounds access to validation data" << std::endl;
    }
    return std::make_pair(inputs[index + splits.first], outputs[index + splits.first]);
}

std::pair<Tensor<double, 3>, Tensor<double, 3>> Dataset::test(int index)
{
    if (index >= inputs.size() - splits.second)
    {
        std::cerr << "out of bounds access to testing data" << std::endl;
    }
    return std::make_pair(inputs[index + splits.second], outputs[index + splits.second]);
}

} // namespace bunji