#pragma once

#include "layer.hpp"

#include <vector>

class Network
{
private:
    std::vector<Layer> layers;
public:
    Network() = default;
    Tensor forward_pass(const Tensor &input);
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives);
    void apply_gradients(double learn_rate);
};