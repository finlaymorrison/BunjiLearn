#pragma once

#include "network.hpp"
#include "dataset.hpp"
#include "loss.hpp"

class Trainer
{
private:
    Network *network;
    Dataset *dataset;
    Loss *loss;
    double learn_rate;
public:
    Trainer(Network *network, Dataset *dataset, Loss *loss, double learn_rate=0.001);
    void train_example(const Tensor &input, const Tensor &expected_output);
    void train_pass();
    void fit(int epochs);
};