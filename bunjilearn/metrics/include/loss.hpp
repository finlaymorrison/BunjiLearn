#pragma once

#include "tensor.hpp"

class Loss
{
private:
public:
    Loss() = default;
    virtual Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) = 0;
    virtual double get_loss(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) = 0;
};

class SquaredError : public Loss
{
private:
public:
    SquaredError() = default;
    Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    double get_loss(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
};

class Crossentropy : public Loss
{
private:
public:
    Crossentropy() = default;
    Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    double get_loss(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
};