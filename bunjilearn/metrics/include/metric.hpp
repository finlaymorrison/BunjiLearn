#include "tensor.hpp"

class Metric
{
private:
public:
    Metric() = default;
    virtual void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) = 0;
    virtual double evaluate() = 0;
};

class Accuracy : public Metric
{
private:
    int correct;
    int total;
public:
    Accuracy();
    void update(const Tensor<double, 3> &output, const Tensor<double, 3> &expected_output) override;
    double evaluate() override;
};