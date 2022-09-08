#include "network.hpp"
#include "log.hpp"

namespace bunji
{

Tensor<double, 3> Network::forward_pass(const Tensor<double, 3> &input, bool training)
{
    Tensor<double, 3> output = input;
    for (Layer *layer : layers)
    {
        output = layer->forward_pass(output, training);
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

void Network::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    std::tuple<std::size_t, std::size_t, std::size_t> next_input = set_input_shape;
    for (Layer *layer : layers)
    {
        if (layer->built)
        {
            if (layer->get_input_shape() != next_input)
            {
                BUNJI_WRN("layer built with incorrect input shape");
                return;
            }
        }
        else
        {
            BUNJI_WRN("({},{},{})", std::get<0>(next_input), std::get<1>(next_input), std::get<2>(next_input));
            layer->build(next_input);
            next_input = layer->get_output_shape();
        }
    }
}

} // namespace bunji
