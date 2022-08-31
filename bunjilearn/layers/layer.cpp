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

} // namespace bunji
