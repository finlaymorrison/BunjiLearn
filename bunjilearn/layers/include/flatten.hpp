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
    Flatten(int d, int h, int w);
    void build(std::size_t x, std::size_t y, std::size_t z) override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input);
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives);

    /* Flatten layer has no parameters */
    void apply_gradients(double learn_rate) override {}
};

} // namespace bunji
