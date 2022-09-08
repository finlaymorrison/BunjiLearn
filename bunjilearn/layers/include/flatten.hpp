#pragma once

#include "layer.hpp"

namespace bunji
{

class Flatten : public Layer
{
private:
    std::size_t x, y, z;
public:
    Flatten();
    Flatten(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void initialize() override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training);
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives);

    /* Flatten layer has no parameters */
    void apply_gradients(__attribute__((unused)) double learn_rate) override {}
};

} // namespace bunji
