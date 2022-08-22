#include "loss.hpp"

Tensor SquaredError::derivative(const Tensor &output, const Tensor &expected_output)
{
    Tensor derivative({{{}}});
    derivative[0][0].reserve(output.size());

    for (int i = 0; i < output.size(); i++)
    {
        derivative[0][0].push_back(output[0][0][i] - expected_output[0][0][i]);
    }

    return derivative;
}