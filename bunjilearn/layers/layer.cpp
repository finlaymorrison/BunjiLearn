#include "layer.hpp"

namespace bunji
{

Layer::Layer() :
    activations()
{}

Tensor<double, 3> Layer::get_activations() const
{
    return activations;
}

std::tuple<std::size_t, std::size_t, std::size_t> Layer::get_output_shape()
{
    return output_shape;
}
std::tuple<std::size_t, std::size_t, std::size_t> Layer::get_input_shape()
{
    return input_shape;
}

} // namespace bunji
