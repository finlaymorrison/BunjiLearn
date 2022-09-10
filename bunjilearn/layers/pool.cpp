#include "pool.hpp"
#include "log.hpp"

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

Tensor<double, 3> MaxPool::forward_pass(const Tensor<double, 3> &input, __attribute__((unused)) bool training)
{
    const auto &[x_in, y_in, z_in] = input_shape;
    const auto &[x_out, y_out, z_out] = get_output_shape();

    for (std::size_t i = 0; i < x_out; ++i)
    {
        for (std::size_t j = 0; j < y_out; ++j)
        {
            for (std::size_t k = 0; k < z_out; ++k)
            {
                activations[i][j][k] = std::numeric_limits<double>::min();
            }
        }
    }

    for (std::size_t i = 0; i < x_in; ++i)
    {
        for (std::size_t j = 0; j < y_in; ++j)
        {
            std::size_t pool_x = i / sx;
            std::size_t pool_y = j / sy;

            for (std::size_t k = 0; k < z_in; ++k)
            {
                if (input[i][j][k] > activations[pool_x][pool_y][k])
                {
                    activations[pool_x][pool_y][k] = input[i][j][k];
                    x_connections[pool_x][pool_y][k] = i;
                    y_connections[pool_x][pool_y][k] = j;
                }
            }
        }
    }
    return activations;
}

Tensor<double, 3> MaxPool::backward_pass(__attribute__((unused)) const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    const auto &[x_in, y_in, z_in] = input_shape;
    const auto &[x_out, y_out, z_out] = get_output_shape();

    Tensor<double, 3>  deriv_inputs({x_in, y_in, z_in});

    for (std::size_t i = 0; i < x_out; ++i)
    {
        for (std::size_t j = 0; j < y_out; ++j)
        {
            for (std::size_t k = 0; k < z_out; ++k)
            {
                std::size_t input_x = x_connections[i][j][k];
                std::size_t input_y = y_connections[i][j][k];
                deriv_inputs[input_x][input_y][k] = output_derivatives[i][j][k];
            }
        }
    }

    return deriv_inputs;
}

} // namespace bunji