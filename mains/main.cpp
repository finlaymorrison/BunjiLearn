#include "dense.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "metric.hpp"
#include "flatten.hpp"
#include "log.hpp"
#include "config.h"

#include <iostream>

int main(int argc, char **argv)
{
    std::cout << "ml_library version " << ml_library_VERSION_MAJOR << "." << ml_library_VERSION_MINOR << '\n' << std::endl;

    BUNJI_DBG("Hello, %d", 42);

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
    
    return 0;
}
