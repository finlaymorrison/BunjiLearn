#include "activation.hpp"

#include <cmath>
#include <iostream>

namespace bunji
{

Activation::Activation() :
    x(0), y(0), z(0)
{
    built = false;
}

Activation::Activation(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    built = false;
    build(set_input_shape);
}

void Activation::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    input_shape = set_input_shape;
    x = std::get<0>(set_input_shape);
    y = std::get<1>(set_input_shape);
    z = std::get<2>(set_input_shape);
    output_shape = std::make_tuple(x, y, z);
    built = true;
}

Tensor<double, 3> ReLU::forward_pass(const Tensor<double, 3> &input)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> output({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = std::max(0.0, input[0][0][i]);
    }

    activations = output;
    return output;
}

Tensor<double, 3> ReLU::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> input_derivatives({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * (input[0][0][i] > 0) ? 1 : 0;
    }

    return input_derivatives;
}



Tensor<double, 3> Sigmoid::forward_pass(const Tensor<double, 3> &input)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> output({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = 1.0 / (1.0 + std::exp(-input[0][0][i]));
    }

    activations = output;
    return output;
}

Tensor<double, 3> Sigmoid::backward_pass(const Tensor<double, 3> &input, const Tensor <double, 3>&output_derivatives)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> input_derivatives({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * activations[0][0][i] * (1 - activations[0][0][i]);
    }

    return input_derivatives;
}



Tensor<double, 3> Tanh::forward_pass(const Tensor<double, 3> &input)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> output({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = std::tanh(input[0][0][i]);
    }

    activations = output;
    return output;
}

Tensor<double, 3> Tanh::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> input_derivatives({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * (1 - activations[0][0][i] * activations[0][0][i]);
    }

    return input_derivatives;
}



Tensor<double, 3> Softmax::forward_pass(const Tensor<double, 3> &input)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> output({1, 1, inputs});

    double sum = 0.0;
    for (int i = 0; i < inputs; i++)
    {
        sum += std::exp(input[0][0][i]);
    }
    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = std::exp(input[0][0][i]) / sum;
    }

    activations = output;
    return output;
}

Tensor<double, 3> Softmax::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    std::size_t inputs = input[0][0].size();
    Tensor<double, 3> input_derivatives({1, 1, inputs});

    for (int i = 0; i < inputs; i++)
    {
        for (int j = 0; j < inputs; j++)
        {
            input_derivatives[0][0][j] += output_derivatives[0][0][i] * activations[0][0][i] * ((j == i ? 1.0 : 0.0) - activations[0][0][j]);
        }
    }
    

    return input_derivatives;
}

} // namespace bunji
