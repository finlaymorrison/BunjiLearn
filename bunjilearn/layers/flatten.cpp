#include "flatten.hpp"

#include <iostream>

namespace bunji
{

Flatten::Flatten(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) :
    Layer()
{
    build(set_input_shape);
}

Flatten::Flatten() :
    Layer(), x(0), y(0), z(0)
{}

void Flatten::initialize()
{
    x = std::get<0>(input_shape);
    y = std::get<1>(input_shape);
    z = std::get<2>(input_shape);
    activations = Tensor<double, 3>({1, 1, x * y * z});
}

Tensor<double, 3> Flatten::forward_pass(const Tensor<double, 3> &input, __attribute__((unused)) bool training)
{
    std::size_t index = 0;
    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                activations[0][0][index++] = input[i][j][k];
            }
        }
    }

    return activations;
}

Tensor<double, 3> Flatten::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({x, y, z});

    int index = 0;
    for (std::size_t i = 0; i < x; ++i)
    {
        for (std::size_t j = 0; j < y; ++j)
        {
            for (std::size_t k = 0; k < z; ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[0][0][index];
                ++index;
            }
        }
    }

    return input_derivatives;
}

} // namespace bunji
