#include "dense.hpp"
#include "log.hpp"

#include <random>

namespace bunji
{

Dense::Dense(std::size_t units) :
    units(units)
{
    built = false;
}

Dense::Dense(std::size_t input, std::size_t units) :
    units(units)
{
    built = false;
    build(std::make_tuple(1, 1, input));
}
void Dense::build(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)
{
    input_shape = set_input_shape;
    if (std::get<0>(input_shape) != 1 || std::get<1>(input_shape) != 1)
    {
        BUNJI_WRN("cannot build dense layer with input shape ({},{},{})", std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape));
        return;
    }
    const std::size_t input = std::get<2>(input_shape);

    weights = Tensor<double, 2>({units, input});
    deriv_weights = Tensor<double, 2>({units, input});
    biases = Tensor<double, 1>({units});
    deriv_biases = Tensor<double, 1>({units});
    
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (auto unit_weights : weights)
    {
        for (double &weight : unit_weights)
        {
            weight = dist(gen);
        }
    }

    output_shape = std::make_tuple(1, 1, units);
    activations = Tensor<double, 3>({1, 1, units});
    
    built = true;
}

Tensor<double, 3> Dense::forward_pass(const Tensor<double, 3> &input)
{
    for (std::size_t i = 0; i < units; ++i)
    {
        /* calculate the dot product of the input and weights */
        double sum = 0.0;
        for (std::size_t j = 0; j < std::get<2>(input_shape); ++j)
        {
            sum += input[0][0][j] * weights[i][j];
        }
        sum += biases[i];

        /* add to the output vector */
        activations[0][0][i] = sum;
    }

    return activations;
}

Tensor<double, 3> Dense::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{    
    /* initialize a tensor for the output */
    Tensor<double, 3> deriv_input({1, 1, std::get<2>(input_shape)});
    
    for (std::size_t i = 0; i < units; ++i)
    {
        for (std::size_t j = 0; j < std::get<2>(input_shape); ++j)
        {
            /* weight derivative calculation */
            deriv_weights[i][j] += input[0][0][j] * output_derivatives[0][0][i];

            /* input derivative calculation */
            deriv_input[0][0][j] += weights[i][j] * output_derivatives[0][0][i];
        }
        
        /* bias derivative calculation */
        deriv_biases[i] += output_derivatives[0][0][i];
    }
    
    return deriv_input;
}

void Dense::apply_gradients(double learn_rate)
{
    for (std::size_t i = 0; i < units; ++i)
    {
        for (std::size_t j = 0; j < std::get<2>(input_shape); ++j)
        {
            weights[i][j] -= deriv_weights[i][j] * learn_rate;
            deriv_weights[i][j] = 0.0;
        }
        biases[i] -= deriv_biases[i] * learn_rate;
        deriv_biases[i] = 0.0;
    }
}

} // namespace bunji
