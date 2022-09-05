#pragma once

#include "layer.hpp"

#include <vector>

namespace bunji
{

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
    void build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
};

} // namespace bunji
