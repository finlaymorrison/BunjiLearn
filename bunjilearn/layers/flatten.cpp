#include "flatten.hpp"

namespace bunji
{

Flatten::Flatten(int d, int h, int w) :
    d(d), h(h), w(w)
{}

Tensor<double, 3> Flatten::forward_pass(const Tensor<double, 3> &input)
{
    Tensor<double, 3> output({1, 1, static_cast<std::size_t>(d * h * w)});

    std::size_t index = 0;
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < w; ++k)
            {
                output[0][0][index++] = input[i][j][k];
            }
        }
    }

    activations = output;
    return output;
}

Tensor<double, 3> Flatten::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> input_derivatives({static_cast<std::size_t>(d), static_cast<std::size_t>(h), static_cast<std::size_t>(w)});

    int index = 0;
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < w; ++k)
            {
                input_derivatives[i][j][k] = output_derivatives[0][0][index];
                ++index;
            }
        }
    }

    return input_derivatives;
}

} // namespace bunji
