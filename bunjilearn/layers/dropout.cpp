#include "dropout.hpp"
#include "log.hpp"

#include <random>

namespace bunji
{

Dropout::Dropout(double rate) :
    Layer(), rate(rate)
{}

Dropout::Dropout(double rate, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) :
    Layer(), rate(rate)
{
    build(set_input_shape);
}

void Dropout::initialize()
{
    auto &[x, y, z] = input_shape;
    connections = Tensor<char, 3>({x, y, z});
    activations = Tensor<double, 3>({x, y, z});
}

Tensor<double, 3> Dropout::forward_pass(const Tensor<double, 3> &input, bool training)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::bernoulli_distribution dist(1.0-rate);

    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                if (training)
                {
                    char connection = dist(gen);
                    activations[i][j][k] = connection * input[i][j][k];
                    connections[i][j][k] = connection;
                }
                else
                {
                    activations[i][j][k] = input[i][j][k] * (1.0 - rate);
                }
            }
        }
    }

    return activations;
}

Tensor<double, 3> Dropout::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> deriv_input({1, 1, std::get<2>(input_shape)});
    
    for (std::size_t i = 0; i < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < std::get<1>(input_shape); ++j)
        {
            for (std::size_t k = 0; k < std::get<2>(input_shape); ++k)
            {
                deriv_input[i][j][k] = connections[i][j][k] * output_derivatives[i][j][k];
            }
        }
    }
    
    return deriv_input;
}

} // namespace bunji