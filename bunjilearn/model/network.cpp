#include "network.hpp"

#include <iostream>

namespace bunji
{

Tensor<double, 3> Network::forward_pass(const Tensor<double, 3> &input)
{
    Tensor<double, 3> output = input;
    for (Layer *layer : layers)
    {
        output = layer->forward_pass(output);
    }
    return output;
}

Tensor<double, 3> Network::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor input_derivatives = output_derivatives;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Tensor<double, 3> layer_input;
        if (i > 0)
        {
            layer_input = layers[i-1]->get_activations();
        }
        else
        {
            layer_input = input;
        }
        input_derivatives = layers[i]->backward_pass(layer_input, input_derivatives);
    }
    return input_derivatives;
}

void Network::apply_gradients(double learn_rate)
{
    for (Layer *layer : layers)
    {
        layer->apply_gradients(learn_rate);
    }
}

void Network::add_layer(Layer *layer)
{
    layers.push_back(layer);
}

} // namespace bunji
