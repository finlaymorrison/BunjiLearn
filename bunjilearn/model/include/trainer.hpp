#pragma once

#include "network.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "metric.hpp"

namespace bunji
{

enum class DatasetPartition
{
    Train = 0,
    Validation = 1,
    Test = 2
};

class Trainer
{
private:
    Network *network;
    Dataset *dataset;
    Loss *loss;
    std::vector<Metric*> metrics;
    double learn_rate;
public:
    Trainer(Network *network, Dataset *dataset, Loss *loss, const std::vector<Metric*> &metrics, double learn_rate=0.001);
    void process_example(const Tensor<double, 3> &input, const Tensor<double, 3> &expected_output, bool training);
    void dataset_pass(DatasetPartition partition, std::size_t dataset_len, std::size_t metrics, bool training);
    void evaluate_metrics(const std::string &part_name, std::size_t dataset_len);
    void fit(std::size_t epochs, std::size_t batch_size);
};

} // namespace bunji
