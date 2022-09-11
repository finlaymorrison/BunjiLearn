#include "dataset.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "metric.hpp"
#include "trainer.hpp"
#include "flatten.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(bunjilearn_py, handle)
{
    handle.doc() = "This is the module docs. Teehee";

    py::class_<bunji::Layer>(
        handle, "PyLayer"
    );

    py::class_<bunji::Activation, bunji::Layer>(
        handle, "PyActivation"
    );

    py::class_<bunji::Metric>(
        handle, "PyMetric"
    );

    py::class_<bunji::Loss, bunji::Metric>(
        handle, "PyLoss"
    );

    py::class_<bunji::Dataset>(
        handle, "PyDataset"
    )
    .def(py::init<std::string>());

    py::class_<bunji::Flatten, bunji::Layer>(
        handle, "PyFlatten"
    )
    .def(py::init<>());

    py::class_<bunji::Dense, bunji::Layer>(
        handle, "PyDense"
    )
    .def(py::init<std::size_t>());

    py::class_<bunji::Sigmoid, bunji::Activation>(
        handle, "PySigmoid"
    )
    .def(py::init<>());

    py::class_<bunji::Softmax, bunji::Activation>(
        handle, "PySoftmax"
    )
    .def(py::init<>());

    py::class_<bunji::Crossentropy, bunji::Loss>(
        handle, "PyCrossentropy"
    )
    .def(py::init<>());

    py::class_<bunji::Accuracy, bunji::Metric>(
        handle, "PyAccuracy"
    )
    .def(py::init<>());

    py::class_<bunji::Network>(
        handle, "PyNetwork"
    )
    .def(py::init<>())
    .def("add_layer", &bunji::Network::add_layer)
    .def("build", &bunji::Network::build);
    
    py::class_<bunji::Trainer>(
        handle, "PyTrainer"
    )
    .def(py::init<bunji::Network*, bunji::Dataset*, bunji::Loss*, std::vector<bunji::Metric*>, double>())
    .def("fit", &bunji::Trainer::fit);
}
