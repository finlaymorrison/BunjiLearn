#include "activation.hpp"

#include <cmath>
#include <iostream>

Tensor ReLU::forward_pass(const Tensor &input)
{
    int inputs = input[0][0].size();
    Tensor output({{{}}});
    output[0][0].resize(inputs);

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = std::max(0.0, input[0][0][i]);
    }

    activations = output;
    return output;
}

Tensor ReLU::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    std::cout << "relu backprop" << std::endl;
    int inputs = input[0][0].size();
    Tensor input_derivatives({{{}}});
    input_derivatives[0][0].resize(inputs);
    std::cout << output_derivatives[0][0].size() << std::endl;

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * (input[0][0][i] > 0) ? 1 : 0;
    }

    return input_derivatives;
}



Tensor Sigmoid::forward_pass(const Tensor &input)
{
    int inputs = input[0][0].size();
    Tensor output({{{}}});
    output[0][0].resize(inputs);

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = 1.0 / (1.0 + std::exp(-input[0][0][i]));
    }

    activations = output;
    return output;
}

Tensor Sigmoid::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    std::cout << "sigmoid backprop" << std::endl;
    int inputs = input[0][0].size();
    Tensor input_derivatives({{{}}});
    input_derivatives[0][0].resize(inputs);
    std::cout << output_derivatives[0][0].size() << std::endl;

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * activations[0][0][i] * (1 - activations[0][0][i]);
    }

    return input_derivatives;
}



Tensor Tanh::forward_pass(const Tensor &input)
{
    int inputs = input[0][0].size();
    Tensor output({{{}}});
    output[0][0].resize(inputs);

    for (int i = 0; i < inputs; i++)
    {
        output[0][0][i] = std::tanh(input[0][0][i]);
    }

    activations = output;
    return output;
}

Tensor Tanh::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    int inputs = input[0][0].size();
    Tensor input_derivatives({{{}}});
    input_derivatives[0][0].resize(inputs);

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = output_derivatives[0][0][i] * (1 - activations[0][0][i] * activations[0][0][i]);
    }

    return input_derivatives;
}



Tensor Softmax::forward_pass(const Tensor &input)
{
    int inputs = input[0][0].size();
    Tensor output({{{}}});
    output[0][0].resize(inputs);

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

Tensor Softmax::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    std::cout << "softmax backprop" << std::endl;
    int inputs = input[0][0].size();
    Tensor input_derivatives({{{}}});
    input_derivatives[0][0].resize(inputs);
    std::cout << output_derivatives[0][0].size() << std::endl;

    std::cout << "\t\t" << inputs << std::endl;

    for (int i = 0; i < inputs; i++)
    {
        input_derivatives[0][0][i] = 0.0;
        for (int j = 0; j < inputs; j++)
        {
            input_derivatives[0][0][i] += activations[0][0][j] * ((j == i ? 1 : 0) - activations[0][0][i]);
        }
    }

    return input_derivatives;
}
