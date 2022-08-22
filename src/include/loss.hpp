#include "tensor.hpp"

class Loss
{
private:
public:
    Loss() = default;
    virtual Tensor derivative(const Tensor &output, const Tensor &expected_output);
};

class SquaredError : public Loss
{
private:
public:
    SquaredError() = default;
    Tensor derivative(const Tensor &output, const Tensor &expected_output) override;
};