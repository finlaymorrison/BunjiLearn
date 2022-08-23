#include "flatten.hpp"

Flatten::Flatten(int d, int h, int w) :
    d(d), h(h), w(w)
{}

Tensor Flatten::forward_pass(const Tensor &input)
{
    Tensor output({{{}}});
    output.reserve(d*h*w);

    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < w; ++k)
            {
                output[0][0].push_back(input[i][j][k]);
            }
        }
    }

    return output;
}

Tensor Flatten::backward_pass(const Tensor &input, const Tensor &output_derivatives)
{
    Tensor input_derivatives({});

    int index = 0;
    for (int i = 0; i < d; ++i)
    {
        input_derivatives.push_back({});
        for (int j = 0; j < h; ++j)
        {
            input_derivatives[i].push_back({});
            for (int k = 0; k < w; ++k)
            {
                input_derivatives[i][j].push_back(output_derivatives[0][0][index]);
                ++index;
            }
        }
    }

    return input_derivatives;
}
