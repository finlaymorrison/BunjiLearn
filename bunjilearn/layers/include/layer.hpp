#pragma once

#include "tensor.hpp"

#include <tuple>

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
    std::tuple<std::size_t, std::size_t, std::size_t> input_shape;
public:
    bool built;
    Layer();

    virtual void build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) = 0;
    std::tuple<std::size_t, std::size_t, std::size_t> get_output_shape();
    std::tuple<std::size_t, std::size_t, std::size_t> get_input_shape();

    virtual Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) = 0;
    virtual Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) = 0;

    Tensor<double, 3> get_activations() const;

    virtual void apply_gradients(double learn_rate) = 0;
};

} // namespace bunji
