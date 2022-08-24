#pragma once

#include "layer.hpp"

class Activation : public Layer
{
private:
public:
    Activation() = default;

    /* Activation layesr have no parameters */
    void apply_gradients(double learn_rate) override {}
    void dump_data() override {}
};

class ReLU : public Activation
{
private:
public:
    ReLU() = default;
    Tensor forward_pass(const Tensor &input) override;
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) override;
};

class Sigmoid : public Activation
{
private:
public:
    Sigmoid() = default;
    Tensor forward_pass(const Tensor &input) override;
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) override;
};

class Softmax : public Activation
{
private:
public:
    Softmax() = default;
    Tensor forward_pass(const Tensor &input) override;
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) override;
};

class Tanh : public Activation
{
private:
public:
    Tanh() = default;
    Tensor forward_pass(const Tensor &input) override;
    Tensor backward_pass(const Tensor &input, const Tensor &output_derivatives) override;
};
