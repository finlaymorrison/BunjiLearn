#pragma once

#include "tensor.hpp"
#include "metric.hpp"

class Loss : public Metric
{
protected:
    double loss;
public:
    Loss();
    
    virtual Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) = 0;

    double evaluate() override;
};

class SquaredError : public Loss
{
private:
public:
    SquaredError() = default;
    Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
};

class Crossentropy : public Loss
{
private:
public:
    Crossentropy() = default;
    Tensor<double, 3> derivative(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
};