#include "loss.hpp"

#include <iostream>
#include <cmath>

Tensor<double, 3> SquaredError::derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    Tensor<double, 3> derivative({1, 1, output[0][0].size()});

    for (int i = 0; i < output[0][0].size(); i++)
    {
        derivative[0][0][i] = output[0][0][i] - expected_output[0][0][i];
    }

    return derivative;
}

double SquaredError::get_loss(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    double loss = 0.0;

    for (int i = 0; i < output[0][0].size(); ++i)
    {
        loss += 0.5 * (output[0][0][i] - expected_output[0][0][i]) * (output[0][0][i] - expected_output[0][0][i]);
    }

    return loss;
}

Tensor<double, 3> Crossentropy::derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    Tensor<double, 3> derivative({1, 1, output[0][0].size()});
    
    for (int i = 0; i < output[0][0].size(); ++i)
    {
        derivative[0][0][i] = -expected_output[0][0][i]*1/(output[0][0][i]*std::log(2));
    }
    
    return derivative;
}

double Crossentropy::get_loss(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    double loss = 0.0;

    for (int i = 0; i < output[0][0].size(); ++i)
    {
        loss -= expected_output[0][0][i] * std::log2(output[0][0][i]);
    }

    return loss;
}
