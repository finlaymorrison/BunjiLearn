#pragma once

#include "layer.hpp"

namespace bunji
{

class Dense : public Layer
{
private:
    std::size_t units;
    Tensor<double, 2> weights;
    Tensor<double, 2> deriv_weights;
    Tensor<double, 1> biases;
    Tensor<double, 1> deriv_biases;
public:
    Dense(std::size_t units);
    Dense(std::size_t inputs, std::size_t units);
    void build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(double learn_rate) override;
};

} // namespace bunji
