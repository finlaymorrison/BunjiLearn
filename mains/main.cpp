#include "dense.hpp"
#include "dataset.hpp"
#include "activation.hpp"
#include "network.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "flatten.hpp"

#include "config.h"

#include <iostream>

int main(int argc, char **argv)
{
    std::cout << "ml_library version " << ml_library_VERSION_MAJOR << "." << ml_library_VERSION_MINOR << '\n' << std::endl;

    Dataset dataset("../scripts/dataset.json");

    Network network;

    Flatten f0(1, 28, 28);
    Dense d0(784, 256);
    Sigmoid a0;
    Dense d1(256, 10);
    Softmax a1;

    network.add_layer(&f0);
    network.add_layer(&d0);
    network.add_layer(&a0);
    network.add_layer(&d1);
    network.add_layer(&a1);

    Crossentropy loss;

    Trainer network_trainer(&network, &dataset, &loss, 5);

    network_trainer.fit(1000000);
    
    return 0;
}
