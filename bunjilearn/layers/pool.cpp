#include "pool.hpp"

#include <limits>

namespace bunji
{

MaxPool::MaxPool(std::size_t sx, std::size_t sy) :
    Layer(), sx(sx), sy(sy)
{}
    
MaxPool::MaxPool(std::size_t sx, std::size_t sy, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape) :
    Layer(), sx(sx), sy(sy)
{
    build(set_input_shape);
}

void MaxPool::initialize()
{
    const auto &[x, y, z] = input_shape;
    std::size_t x_out = x / sx + (x % sx > 0);
    std::size_t y_out = y / sy + (y % sy > 0);
    activations = Tensor<double, 3>({x_out, y_out, z});
    x_connections = Tensor<std::size_t, 3>({x_out, y_out, z});
    y_connections = Tensor<std::size_t, 3>({x_out, y_out, z});
}

void MaxPool::calculate_pool(const Tensor<double, 3> &input, std::size_t pool_x, std::size_t pool_y, std::size_t pool_z)
{
    double max = std::numeric_limits<double>::min();
    std::size_t max_i=0, max_j=0;
    for (std::size_t i = 0; i < pool_x && (pool_x + i) < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < pool_y && (pool_y + j) < std::get<1>(input_shape); ++j)
        {
            double value = input[pool_x + i][pool_y + j][pool_z];
            if (value > max)
            {
                max = value;
                max_i = i;
                max_j = j;
            }
        }
    }
    x_connections[pool_x][pool_y][pool_z] = max_i;
    y_connections[pool_x][pool_y][pool_z] = max_j;
    activations[pool_x][pool_y][pool_z] = max;
}

Tensor<double, 3> MaxPool::forward_pass(const Tensor<double, 3> &input, __attribute__((unused)) bool training)
{
    const auto &[x_out, y_out, z_out] = get_output_shape();
    for (std::size_t i = 0; i < x_out; ++i)
    {
        for (std::size_t j = 0; j < y_out; ++j)
        {
            for (std::size_t k = 0; k < z_out; ++k)
            {
                calculate_pool(input, i*sx, j*sy, k);
            }
        }
    }

    return activations;
}

void MaxPool::backpropagate_pool(Tensor<double, 3> &deriv_input, double output_derivative, std::size_t pool_x, std::size_t pool_y, std::size_t pool_z)
{
    for (std::size_t i = 0; i < pool_x && (pool_x + i) < std::get<0>(input_shape); ++i)
    {
        for (std::size_t j = 0; j < pool_y && (pool_y + j) < std::get<1>(input_shape); ++j)
        {
            if (x_connections[pool_x][pool_y][pool_z] == i && y_connections[pool_x][pool_y][pool_z] == j)
            {
                deriv_input[pool_x + i][pool_y + j][pool_z] = output_derivative;
            }
            else
            {
                deriv_input[pool_x + i][pool_y + j][pool_z] = 0.0;
            }
        }
    }
}

Tensor<double, 3> MaxPool::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    Tensor<double, 3> deriv_input({std::get<0>(input_shape), std::get<1>(input_shape), std::get<2>(input_shape)});

    const auto &[x_out, y_out, z_out] = get_output_shape();
    for (std::size_t i = 0; i < x_out; ++i)
    {
        for (std::size_t j = 0; j < y_out; ++j)
        {
            for (std::size_t k = 0; k < z_out; ++k)
            {
                backpropagate_pool(deriv_input, output_derivatives[i][j][k], i, j, k);
            }
        }
    }
    
    return deriv_input;
}

} // namespace bunji