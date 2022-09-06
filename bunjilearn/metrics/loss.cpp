#include "loss.hpp"
#include "log.hpp"

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
    const auto &[x, y, z] = output.shape();
    Tensor<double, 3> derivative({x, y, z});

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                derivative[i][j][k] = output[i][j][k] - expected_output[i][j][k];
            }
        }
    }

    return derivative;
}

void SquaredError::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    const auto &[x, y, z] = output.shape();

    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                loss += (output[i][j][k] - expected_output[i][j][k]) * (output[i][j][k] - expected_output[i][j][k]);
            }
        }
    }
}

Tensor<double, 3> Crossentropy::derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    const auto &[x, y, z] = output.shape();
    Tensor<double, 3> derivative({x, y, z});
    
    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                derivative[i][j][k] = -expected_output[i][j][k]*1/(output[i][j][k]*std::log(2));
            }
        }
    }
    
    return derivative;
}

void Crossentropy::update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output)
{
    const auto &[x, y, z] = output.shape();
    
    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                loss -= expected_output[i][j][k] * std::log2(output[i][j][k]);
            }
        }
    }
}

} // namespace bunji
