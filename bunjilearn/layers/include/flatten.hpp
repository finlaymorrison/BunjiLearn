#pragma once

#include "layer.hpp"

namespace bunji
{

class Flatten : public Layer
{
private:
    int d, h, w;
public:
    Flatten();
    Flatten(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input);
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives);

    /* Flatten layer has no parameters */
    void apply_gradients(double learn_rate) override {}
};

} // namespace bunji
