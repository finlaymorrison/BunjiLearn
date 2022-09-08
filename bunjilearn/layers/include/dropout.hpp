#pragma once

#include "layer.hpp"

namespace bunji
{

class Dropout : public Layer
{
private:
    std::size_t x, y, z;
    Tensor<char, 3> connections;
    double rate;
public:
    Dropout(double rate);
    Dropout(double rate, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(__attribute__((unused)) double learn_rate) override {};
};

} // namespace bunji
