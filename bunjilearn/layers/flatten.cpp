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
    d(0), h(0), w(0)
{
    built = false;
}

void Flatten::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    input_shape = set_input_shape;
    d = std::get<0>(set_input_shape);
    h = std::get<1>(set_input_shape);
    w = std::get<2>(set_input_shape);
    output_shape = std::make_tuple(1, 1, d * h * w);
    activations = Tensor<double, 3>({1, 1, d * h * w});
    built = true;
}

Tensor<double, 3> Flatten::forward_pass(const Tensor<double, 3> &input)
{
    std::size_t index = 0;
    for (std::size_t i = 0; i < d; ++i)
    {
        for (std::size_t j = 0; j < h; ++j)
        {
            for (std::size_t k = 0; k < w; ++k)
            {
                activations[0][0][index++] = input[i][j][k];
            }
        }
    }

    return activations;
}

Tensor<double, 3> Flatten::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({static_cast<std::size_t>(d), static_cast<std::size_t>(h), static_cast<std::size_t>(w)});

    int index = 0;
    for (std::size_t i = 0; i < d; ++i)
    {
        for (std::size_t j = 0; j < h; ++j)
        {
            for (std::size_t k = 0; k < w; ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[0][0][index];
                ++index;
            }
        }
    }

    return input_derivatives;
}

} // namespace bunji
