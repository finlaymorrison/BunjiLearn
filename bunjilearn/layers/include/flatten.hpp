#pragma once

#include "layer.hpp"

class Flatten : public Layer
{
private:
    int d, h, w;
public:
    Flatten(int d, int h, int w);
    Tensor forward_pass(const Tensor &input);
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives);

    /* Flatten layer has no parameters */
    void apply_gradients(double learn_rate) override {}
    void dump_data() override {}
};