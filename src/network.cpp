#include "network.hpp"

Tensor Network::forward_pass(const Tensor &input)
{
    Tensor output = input;
    for (Layer &layer : layers)
    {
        output = layer.forward_pass(output);
    }
    return output;
}

Tensor Network::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    Tensor input_derivatives = output_derivatives;
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        input_derivatives = layers[i].backward_pass(input, input_derivatives);
    }
    return input_derivatives;
}

void Network::apply_gradients(double learn_rate)
{
    for (Layer &layer : layers)
    {
        layer.apply_gradients(learn_rate);
    }
}