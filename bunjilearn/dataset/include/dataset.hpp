#pragma once

#include "tensor.hpp"

#include <vector>
#include <utility>
#include <string>

class Dataset
{
private:
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::pair<int, int> splits;

    void parse_data(const std::string &data);
public:
    Dataset(const std::string &filepath);
    int train_len();
    int val_len();
    int test_len();
    std::pair<Tensor, Tensor> train(int index);
    std::pair<Tensor, Tensor> val(int index);
    std::pair<Tensor, Tensor> test(int index);
};