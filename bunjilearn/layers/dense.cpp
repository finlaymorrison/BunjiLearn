#include "dense.hpp"

#include <random>
#include <iostream>

namespace bunji
{

Dense::Dense(int input, int units) :
    weights(units, std::vector<double>(input,0.0)),
    deriv_weights(units, std::vector<double>(input,0.0)),
    biases(units, 0.0),
    deriv_biases(units, 0.0)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (std::vector<double> &unit_weights : weights)
    {
        for (double &weight : unit_weights)
        {
            weight = dist(gen);
        }
    }
}

Tensor<double, 3> Dense::forward_pass(const Tensor<double, 3> &input)
{
    const std::size_t units = weights.size();
    const std::size_t input_size = input[0][0].size();

    /* initialize a tensor for the output */
    Tensor<double, 3> output({1, 1, units});

    for (int i = 0; i < units; ++i)
    {
        /* calculate the dot product of the input and weights */
        double sum = 0.0;
        for (int j = 0; j < input_size; ++j)
        {
            sum += input[0][0][j] * weights[i][j];
        }
        sum += biases[i];

        /* add to the output vector */
        output[0][0][i] = sum;
    }

    activations = output;
    return output;
}

Tensor<double, 3> Dense::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    const std::size_t units = weights.size();
    const std::size_t input_size = input[0][0].size();
    const std::size_t output_size = output_derivatives[0][0].size();

    /* checking that the output derivatives are a valid size */
    if (output_size != units)
    {
        std::cerr << "output size does not match units" << std::endl;
        return Tensor<double, 3>({1, 1, 1});
    }
    
    /* initialize a tensor for the output */
    Tensor<double, 3> deriv_input({1, 1, input_size});
    
    for (int i = 0; i < units; ++i)
    {
        for (int j = 0; j < input_size; ++j)
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
    for (int i = 0; i < weights.size(); ++i)
    {
        for (int j = 0; j < weights[i].size(); ++j)
        {
            weights[i][j] -= deriv_weights[i][j] * learn_rate;
            deriv_weights[i][j] = 0.0;
        }
    }

    for (int i = 0; i < biases.size(); ++i)
    {
        biases[i] -= deriv_biases[i] * learn_rate;
        deriv_biases[i] = 0.0;
    }
}

void Dense::dump_data()
{
    for (int i = 0; i < weights.size(); ++i)
    {
        std::cout << i << ":";
        for (const double weight : weights[i])
        {
            std::cout << weight << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

} // namespace bunji
