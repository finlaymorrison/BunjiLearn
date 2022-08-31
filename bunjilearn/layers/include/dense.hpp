#pragma once

#include "layer.hpp"

namespace bunji
{

class Dense : public Layer
{
private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> deriv_weights;
    std::vector<double> biases;
    std::vector<double> deriv_biases;
public:
    Dense(int inputs, int units);
    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(double learn_rate) override;

    void dump_data() override;
};

} // namespace bunji
