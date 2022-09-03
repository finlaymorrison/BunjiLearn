#pragma once

#include "tensor.hpp"

namespace bunji
{

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
    Tensor<double, 3> activations;
public:
    Layer();

    virtual Tensor<double, 3> forward_pass(const Tensor<double, 3> &input) = 0;
    virtual Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) = 0;

    Tensor<double, 3> get_activations() const;

    virtual void apply_gradients(double learn_rate) = 0;
};

} // namespace bunji
