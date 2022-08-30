#include "layer.hpp"

Layer::Layer() :
    activations()
{}

Tensor<double, 3> Layer::get_activations() const
{
    return activations;
}