#include "trainer.hpp"
#include "log.hpp"

namespace bunji
{

Trainer::Trainer(Network *network, Dataset *dataset, Loss *loss, const std::vector<Metric*> &metrics, double learn_rate) :
    network(network), dataset(dataset), loss(loss), metrics(metrics), learn_rate(learn_rate)
{}

void Trainer::process_example(const Tensor<double, 3> &input, const Tensor<double, 3> &expected_output, bool training)
{
    Tensor<double, 3> output = network->forward_pass(input);

    for (Metric *metric : metrics)
    {
        metric->update(output, expected_output);
    }

    if (training)
    {
        Tensor<double, 3> output_derivatives = loss->derivative(output, expected_output);
        Tensor<double, 3> input_derivatives = network->backward_pass(input, output_derivatives);
    }
}

void Trainer::dataset_pass(DatasetPartition partition, std::size_t dataset_len, std::size_t batch_size, bool training)
{
    for (std::size_t i = 0; i < dataset_len; ++i)
    {
        std::pair<Tensor<double, 3>, Tensor<double, 3>> example;

        switch(partition)
        {
        case DatasetPartition::Train: example = dataset->train(i);
            break;
        case DatasetPartition::Validation: example = dataset->val(i);
            break;
        case DatasetPartition::Test: example = dataset->test(i);
            break;
        }

        process_example(example.first, example.second, training);

        if (training && ((i+1) % 32 == 0))
        {
            network->apply_gradients(learn_rate / batch_size);
        }
    }
    if (training && (dataset_len % batch_size != 0))
    {
        network->apply_gradients(learn_rate / (dataset_len % batch_size));
    }
}

void Trainer::evaluate_metrics(const std::string &part_name, std::size_t dataset_len)
{
    BUNJI_PRINT("\t{} : ", part_name);
    for (Metric *metric : metrics)
    {
        double metric_value = metric->evaluate(dataset_len);
        BUNJI_PRINT("\t{}:{}", metric->get_name(), metric_value);
    }
    BUNJI_PRINT("\n");
}

void Trainer::fit(std::size_t epochs, std::size_t batch_size)
{
    for (std::size_t i = 0; i < epochs; ++i)
    {
        BUNJI_PRINT("Epoch: {}\n", i);

        dataset_pass(DatasetPartition::Train, dataset->train_len(), batch_size, true);
        evaluate_metrics("Training Data  ", dataset->train_len());

        dataset_pass(DatasetPartition::Validation, dataset->val_len(), batch_size, false);
        evaluate_metrics("Validation Data", dataset->val_len());
    }
    dataset_pass(DatasetPartition::Test, dataset->test_len(), batch_size, false);
    evaluate_metrics("Testing Data   ", dataset->test_len());
}

} // namespace bunji
