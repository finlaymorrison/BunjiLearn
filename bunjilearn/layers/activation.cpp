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
    activations = Tensor<double, 3>({x, y, z});
    built = true;
}

Tensor<double, 3> ReLU::forward_pass(const Tensor<double, 3> &input)
{
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                activations[i][j][k] = std::max(0.0, input[i][j][k]);
            }
        }
    }
    
    return activations;
}

Tensor<double, 3> ReLU::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape)});

    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[i][j][k] && (input[i][j][k] > 0);
            }
        }
    }

    return input_derivatives;
}



Tensor<double, 3> Sigmoid::forward_pass(const Tensor<double, 3> &input)
{
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                activations[i][j][k] = 1.0 / (1.0 + std::exp(-input[i][j][k]));
            }
        }
    }
    
    return activations;
}

Tensor<double, 3> Sigmoid::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor <double, 3>&output_derivatives)
{
    Tensor<double, 3> input_derivatives({std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape)});

    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[i][j][k] * activations[i][j][k] * (1 - activations[i][j][k]);
            }
        }
    }

    return input_derivatives;
}



Tensor<double, 3> Tanh::forward_pass(const Tensor<double, 3> &input)
{
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                activations[i][j][k] = std::tanh(input[i][j][k]);
            }
        }
    }
    
    return activations;
}

Tensor<double, 3> Tanh::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape)});

    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[i][j][k] * (1 - activations[i][j][k] * activations[i][j][k]);
            }
        }
    }

    return input_derivatives;
}


/*
 * It doesnt really makes sense to use softmax on an input which is more
 * than 1 dimension, other than potentially trying to estimate many random
 * variables, and so each vector in the deepest axis will be considered
 * as a separate random variable.
 */
Tensor<double, 3> Softmax::forward_pass(const Tensor<double, 3> &input)
{
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                sum += std::exp(input[i][j][k]);
            }
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                activations[i][j][k] = std::exp(input[i][j][k]) / sum;
            }
        }
    }

    return activations;
}

Tensor<double, 3> Softmax::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape)});
    
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); k++)
            {
                for (std::size_t l = 0; l < std::get<2>(input_shape); l++)
                {
                    input_derivatives[i][j][l] += output_derivatives[i][j][k] * activations[i][j][k] * ((l == k ? 1.0 : 0.0) - activations[i][j][l]);
                }
            }
        }
    }

    return input_derivatives;
}

} // namespace bunji
