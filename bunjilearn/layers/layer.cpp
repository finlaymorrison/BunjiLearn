#include "layer.hpp"

namespace bunji
{

Layer::Layer() :
    activations(), built(false)
{}

void Layer::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    input_shape = set_input_shape;
    initialize();
    built = true;
}

Tensor<double, 3> Layer::get_activations() const
{
    return activations;
}

std::tuple<std::size_t, std::size_t, std::size_t> Layer::get_output_shape()
{
    return activations.shape();
}
std::tuple<std::size_t, std::size_t, std::size_t> Layer::get_input_shape()
{
    return input_shape;
}

} // namespace bunji
