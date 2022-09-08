#include "convolution.hpp"

#include <random>

namespace bunji
{

Convolution::Convolution(std::size_t kernel_count, std::size_t kx, std::size_t ky, std::size_t sx, std::size_t sy) :
    Layer(), kernel_count(kernel_count), kx(kx), ky(ky), sx(sx), sy(sy)
{}

    
Convolution::Convolution(std::size_t kernel_count, std::size_t kx, std::size_t ky, std::size_t sx, std::size_t sy, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape)  :
    Layer(), kernel_count(kernel_count), kx(kx), ky(ky), sx(sx), sy(sy)
{
    build(set_input_shape);
}
    
void Convolution::initialize()
{
    const auto &[x, y, z] = input_shape;
    
    kernels = Tensor<double, 4>({kernel_count, kx, ky, z});
    deriv_kernels = Tensor<double, 4>({kernel_count, kx, ky, z});
    biases = Tensor<double, 1>({kernel_count});
    deriv_biases = Tensor<double, 1>({kernel_count});
    
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t i = 0; i < kernel_count; ++i)
    {
        for (std::size_t j = 0; j < kx; ++j)
        {
            for (std::size_t k = 0; k < ky; ++k)
            {
                for (std::size_t l = 0; l < z; ++l)
                {
                    kernels[i][k][k][l] = dist(gen);
                }
            }
        }
    }

    std::size_t x_out = (x - kx) / sx;
    std::size_t y_out = (y - ky) / sy;
    activations = Tensor<double, 3>({x_out, y_out, kernel_count});
}

Tensor<double, 3> Convolution::forward_pass(const Tensor<double, 3> &input, __attribute__((unused)) bool training)
{
    const auto &[x, y, z] = input_shape;
    
    for (std::size_t i = 0; i < kernel_count; ++i)
    {
        for (std::size_t j = 0; j < x-kx; j += sx)
        {
            for (std::size_t k = 0; k < y-ky; k += sy)
            {
                activations[i][j][k] = biases[i];
                for (std::size_t l = 0; l < kx; ++l)
                {
                    for (std::size_t m = 0; m < ky; ++m)
                    {
                        for (std::size_t n = 0; n < z; ++n)
                        {
                            activations[j][k][i] += kernels[i][l][m][n] * input[j+l][k+m][n];
                        }
                    }
                }
            }
        }
    }

    return activations;
}

Tensor<double, 3> Convolution::backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives)
{
    const auto &[x, y, z] = input_shape;
    Tensor<double, 3> input_derivatives({x, y, z});
    
    for (std::size_t i = 0; i < kernel_count; ++i)
    {
        for (std::size_t j = 0; j < x-kx; j += sx)
        {
            for (std::size_t k = 0; k < y-ky; k += sy)
            {
                deriv_biases[i] += output_derivatives[i][j][k];
                for (std::size_t l = 0; l < kx; ++l)
                {
                    for (std::size_t m = 0; m < ky; ++m)
                    {
                        for (std::size_t n = 0; n < z; ++n)
                        {
                            deriv_kernels[i][l][m][n] += output_derivatives[i][j][j] * input[j+l][k+m][n];
                            input_derivatives[j+l][k+m][n] += output_derivatives[i][j][j] * kernels[i][l][m][n];
                        }
                    }
                }
            }
        }
    }

    return input_derivatives;
}

void Convolution::apply_gradients(double learn_rate)
{
    for (std::size_t i = 0; i < kernel_count; ++i)
    {
        for (std::size_t j = 0; j < kx; ++j)
        {
            for (std::size_t k = 0; k < ky; ++k)
            {
                for (std::size_t l = 0; l < std::get<2>(input_shape); ++l)
                {
                    kernels[i][k][k][l] -= learn_rate * deriv_kernels[i][k][k][l];
                }
            }
        }
        biases[i] -= learn_rate * deriv_biases[i];
    }
}

} // namespace bunji
