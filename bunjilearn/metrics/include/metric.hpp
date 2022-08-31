#pragma once

#include "tensor.hpp"

#include <string>

namespace bunji
{

class Metric
{
private:
public:
    Metric() = default;
    virtual void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) = 0;
    virtual double evaluate(int example_count) = 0;
    virtual std::string get_name() = 0;
};

class Accuracy : public Metric
{
private:
    int correct;
public:
    Accuracy();
    void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    double evaluate(int example_count) override;
    std::string get_name() override;
};

} // namespace bunji
