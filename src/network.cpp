#include "network.hpp"

#include <iostream>

Tensor Network::forward_pass(const Tensor &input)
{
    Tensor output = input;
    for (Layer *layer : layers)
    {
        output = layer->forward_pass(output);
    }
    return output;
}

Tensor Network::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    Tensor input_derivatives = output_derivatives;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        std::cout << input_derivatives[0][0].size() << std::endl;
        input_derivatives = layers[i]->backward_pass(input, input_derivatives);
    }
    std::cout << input_derivatives[0][0].size() << std::endl;
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