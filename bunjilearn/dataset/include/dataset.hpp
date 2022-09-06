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
    std::pair<std::size_t, std::size_t> splits;

    void parse_data(const std::string &data);
public:
    Dataset(const std::string &filepath);
    std::size_t train_len();
    std::size_t val_len();
    std::size_t test_len();
    std::pair<Tensor<double, 3>, Tensor<double, 3>> train(std::size_t index);
    std::pair<Tensor<double, 3>, Tensor<double, 3>> val(std::size_t index);
    std::pair<Tensor<double, 3>, Tensor<double, 3>> test(std::size_t index);
};

} // namespace bunji
