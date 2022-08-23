#pragma once

#include "tensor.hpp"

/* 
 * general class for a layer in a neural network. The forward and backward
 * functions must be overidden in a child class to implement the
 * functionality of the layer. Layers take 3 dimensional vectors as both an
 * input and an output.
 */
class Layer
{
private:
protected:
    Tensor activations;
public:
    Layer();

    virtual Tensor forward_pass(const Tensor &input) = 0;
    virtual Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) = 0;

    Tensor get_activations() const;

    virtual void apply_gradients(double learn_rate) = 0;

    virtual void dump_data() = 0;
};