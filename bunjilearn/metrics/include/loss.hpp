#pragma once

#include "tensor.hpp"

class Loss
{
private:
public:
    Loss() = default;
    virtual Tensor derivative(const Tensor &output, const Tensor &expected_output) = 0;
    virtual double get_loss(const Tensor &output, const Tensor &expected_output) = 0;
};

class SquaredError : public Loss
{
private:
public:
    SquaredError() = default;
    Tensor derivative(const Tensor &output, const Tensor &expected_output) override;
    double get_loss(const Tensor &output, const Tensor &expected_output) override;
};

class Crossentropy : public Loss
{
private:
public:
    Crossentropy() = default;
    Tensor derivative(const Tensor &output, const Tensor &expected_output) override;
    double get_loss(const Tensor &output, const Tensor &expected_output) override;
};