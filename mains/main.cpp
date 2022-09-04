#include "dense.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "metric.hpp"
#include "flatten.hpp"
#include "log.hpp"
#include "tensor.hpp"
#include "config.h"

#include <iostream>

int main(int argc, char **argv)
{
    BUNJI_INF("ml_library version {}.{}", ml_library_VERSION_MAJOR, ml_library_VERSION_MINOR);

    bunji::Tensor<int, 2> tensor({10, 10});

    int index = 0;

    for (auto vector : tensor)
    {
        for (auto &value : vector)
        {
            value = index++;
        }
    }
    for (auto vector : tensor)
    {
        for (auto value : vector)
        {
            std::cout << value << std::endl;
        }
    }

    /*
    bunji::Tensor<int, 2>::iterator it_1;
    for (it_1 = tensor.begin(); it_1 != tensor.end(); ++it_1)
    {
        bunji::TensorView<int, 1>::iterator it_2;
        for (it_2 = (*it_1).begin(); it_2 != (*it_1).end(); ++it_2)
        {
            *it_2 = index++;
        }
    }
    for (it_1 = tensor.begin(); it_1 != tensor.end(); ++it_1)
    {
        bunji::TensorView<int, 1>::iterator it_2;
        for (it_2 = (*it_1).begin(); it_2 != (*it_1).end(); ++it_2)
        {
            std::cout << *it_2 << std::endl; 
        }
    }
    */

    /*
    bunji::Dataset dataset("../scripts/dataset.json");

    bunji::Network network;

    bunji::Flatten f0(1, 28, 28);
    bunji::Dense d0(784, 256);
    bunji::Sigmoid a0;
    bunji::Dense d1(256, 10);
    bunji::Softmax a1;

    network.add_layer(&f0);
    network.add_layer(&d0);
    network.add_layer(&a0);
    network.add_layer(&d1);
    network.add_layer(&a1);

    bunji::Crossentropy loss;

    bunji::Crossentropy loss_metric;
    bunji::Accuracy acc_metric;
    
    std::vector<bunji::Metric*> metrics = {&loss_metric, &acc_metric};

    bunji::Trainer network_trainer(&network, &dataset, &loss, metrics, 5);

    network_trainer.fit(1000000);
    */
    return 0;
}
