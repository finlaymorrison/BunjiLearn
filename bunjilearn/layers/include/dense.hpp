#pragma once

#include "layer.hpp"

namespace bunji
{

class Dense : public Layer
{
private:
    int units;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> deriv_weights;
    std::vector<double> biases;
    std::vector<double> deriv_biases;
public:
    Dense(int units);
    Dense(int inputs, int units);
    void build(std::size_t x, std::size_t y, std::size_t z) override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(double learn_rate) override;
};

} // namespace bunji
