#include "flatten.hpp"

#include <iostream>

namespace bunji
{

Flatten::Flatten(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    built = false;
    build(set_input_shape);
}

Flatten::Flatten() :
    x(0), y(0), z(0)
{
    built = false;
}

void Flatten::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    input_shape = set_input_shape;
    x = std::get<0>(set_input_shape);
    y = std::get<1>(set_input_shape);
    z = std::get<2>(set_input_shape);
    output_shape = std::make_tuple(1, 1, x * y * z);
    activations = Tensor<double, 3>({1, 1, x * y * z});
    built = true;
}

Tensor<double, 3> Flatten::forward_pass(const Tensor<double, 3> &input)
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

Tensor<double, 3> Flatten::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
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
