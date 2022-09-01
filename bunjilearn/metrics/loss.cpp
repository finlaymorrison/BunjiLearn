#include "loss.hpp"

#include <cmath>

namespace bunji
{

Loss::Loss() :
    loss(0.0)
{}

double Loss::evaluate(int example_count)
{
    double result = loss / example_count;
    loss = 0.0;
    return result;
}

std::string Loss::get_name()
{
    return std::string("loss");
}

Tensor<double, 3> SquaredError::derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    Tensor<double, 3> derivative({1, 1, output[0][0].size()});

    for (int i = 0; i < output[0][0].size(); i++)
    {
        derivative[0][0][i] = output[0][0][i] - expected_output[0][0][i];
    }

    return derivative;
}

void SquaredError::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    for (int i = 0; i < output[0][0].size(); ++i)
    {
        loss += 0.5 * (output[0][0][i] - expected_output[0][0][i]) * (output[0][0][i] - expected_output[0][0][i]);
    }
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

void Crossentropy::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    for (int i = 0; i < output[0][0].size(); ++i)
    {
        loss -= expected_output[0][0][i] * std::log2(output[0][0][i]);
    }
}

} // namespace bunji
