#pragma once

#include "layer.hpp"

#include <vector>

class Network
{
private:
    std::vector<Layer*> layers;
public:
    Network() = default;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input);
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives);
    void apply_gradients(double learn_rate);

    void add_layer(Layer *layer);
};