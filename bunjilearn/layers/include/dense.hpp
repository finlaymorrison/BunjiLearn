#pragma once

#include "layer.hpp"

class Dense : public Layer
{
private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> deriv_weights;
    std::vector<double> biases;
    std::vector<double> deriv_biases;
public:
    Dense(int inputs, int units);
    Tensor forward_pass(const Tensor &input) override;
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) override;

    void apply_gradients(double learn_rate) override;

    void dump_data() override;
};