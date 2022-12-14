#pragma once

#include "layer.hpp"

namespace bunji
{

class MaxPool : public Layer
{
private:
    std::size_t x, y, z;
    std::size_t sx, sy;
    Tensor<std::size_t, 3> x_connections;
    Tensor<std::size_t, 3> y_connections;
public:
    MaxPool(std::size_t sx, std::size_t sy);
    MaxPool(std::size_t sx, std::size_t sy, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void initialize() override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(__attribute__((unused)) double learn_rate) override {};
};

} // namespace bunji
