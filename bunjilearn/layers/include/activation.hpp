#pragma once

#include "layer.hpp"

namespace bunji
{

class Activation : public Layer
{
private:
protected:
    std::size_t x, y, z;
public:
    Activation();
    Activation(std::tuple<std::size_t, std::size_t, std::size_t> set_input_shape);
    void initialize() override;

    /* Activation layers have no parameters */
    void apply_gradients(__attribute__((unused)) double learn_rate) override {}
};

class ReLU : public Activation
{
private:
public:
    ReLU() = default;
    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;
};

class Sigmoid : public Activation
{
private:
public:
    Sigmoid() = default;
    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;
};

class Softmax : public Activation
{
private:
public:
    Softmax() = default;
    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;
};

class Tanh : public Activation
{
private:
public:
    Tanh() = default;
    Tensor<double, 3> forward_pass(const Tensor<double, 3> &input, bool training) override;
    Tensor<double, 3> backward_pass(const Tensor<double, 3> &input, const Tensor<double, 3> &output_derivatives) override;
};

} // namespace bunji
