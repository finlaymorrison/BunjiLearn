#pragma once

#include "tensor.hpp"

#include <vector>
#include <utility>
#include <string>

namespace bunji
{

class Dataset
{
private:
    std::vector<Tensor<double, 3>> inputs;
    std::vector<Tensor<double, 3>> outputs;
    std::pair<int, int> splits;

    void parse_data(const std::string &data);
public:
    Dataset(const std::string &filepath);
    int train_len();
    int val_len();
    int test_len();
    std::pair<Tensor<double, 3>, Tensor<double, 3>> train(int index);
    std::pair<Tensor<double, 3>, Tensor<double, 3>> val(int index);
    std::pair<Tensor<double, 3>, Tensor<double, 3>> test(int index);
};

} // namespace bunji
