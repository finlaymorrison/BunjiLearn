#include "trainer.hpp"

#include <iostream>

Trainer::Trainer(Network *network, Dataset *dataset, Loss *loss, double learn_rate) :
    network(network), dataset(dataset), loss(loss), learn_rate(learn_rate)
{}

double Trainer::train_example(const Tensor &input, const Tensor &expected_output)
{
    Tensor output = network->forward_pass(input);
    double network_loss = loss->get_loss(output, expected_output);
    Tensor output_derivatives = loss->derivative(output, expected_output);
    Tensor input_derivatives = network->backward_pass(input, output_derivatives);

    return network_loss;
}

double Trainer::train_pass()
{
    double total_loss = 0.0;
    for (int i = 0; i < dataset->train_len(); ++i)
    {
        std::pair<Tensor, Tensor> example = dataset->train(i);
        total_loss += train_example(example.first, example.second);
    }
    network->apply_gradients(learn_rate / dataset->train_len());
    return total_loss / dataset->train_len();
}

void Trainer::fit(int epochs)
{
    for (int i = 0; i < epochs; ++i)
    {
        double network_loss = train_pass();
        std::cout << "Epoch: " << i << " Loss: " << network_loss << std::endl;
    }
}