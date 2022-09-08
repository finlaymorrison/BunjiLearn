#pragma once

#include "layer.hpp"

namespace bunji
{

class Convolution : public Layer
{
private:
    Tensor<double, 4> kernels;
    Tensor<double, 4> deriv_kernels;
    Tensor<double, 1> biases;
    Tensor<double, 1> deriv_biases;
    std::size_t kernel_count;
    std::size_t kx, ky;
    std::size_t sx, sy;
public:
    Convolution(std::size_t kernel_count, std::size_t kx, std::size_t ky, std::size_t sx, std::size_t sy);
    Convolution(std::size_t kernel_count, std::size_t kx, std::size_t ky, std::size_t sx, std::size_t sy, std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void initialize() override;

    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;

    void apply_gradients(double learn_rate) override;
};

} // namespace bunji