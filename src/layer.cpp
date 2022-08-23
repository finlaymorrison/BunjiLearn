#include "layer.hpp"

Layer::Layer() :
    activations()
{}

Tensor Layer::get_activations() const
{
    return activations;
}