#include "loss.hpp"

#include <iostream>

Tensor SquaredError::derivative(const Tensor &output, const Tensor &expected_output)
{
    Tensor derivative({{{}}});
    derivative[0][0].reserve(output[0][0].size());

    for (int i = 0; i < output[0][0].size(); i++)
    {
        derivative[0][0].push_back(output[0][0][i] - expected_output[0][0][i]);
    }

    return derivative;
}

double SquaredError::get_loss(const Tensor &output, const Tensor &expected_output)
{
    double loss = 0.0;

    for (int i = 0; i < output[0][0].size(); ++i)
    {
        loss += 0.5 * (output[0][0][i] - expected_output[0][0][i]) * (output[0][0][i] - expected_output[0][0][i]);
    }

    return loss;
}