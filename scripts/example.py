import bunjilearn_py

dataset = bunjilearn_py.PyDataset("dataset.json")

flatten_0 = bunjilearn_py.PyFlatten()
dense_0 = bunjilearn_py.PyDense(256)
activation_0 = bunjilearn_py.PySigmoid()
dense_1 = bunjilearn_py.PyDense(10)
activation_1 = bunjilearn_py.PySoftmax()

loss = bunjilearn_py.PyCrossentropy()

metric_0 = bunjilearn_py.PyCrossentropy()
metric_1 = bunjilearn_py.PyAccuracy()

metrics = [metric_0, metric_1]

network = bunjilearn_py.PyNetwork()
network.add_layer(flatten_0)
network.add_layer(dense_0)
network.add_layer(activation_0)
network.add_layer(dense_1)
network.add_layer(activation_1)
network.build((28, 28, 1))

trainer = bunjilearn_py.PyTrainer(network, dataset, loss, metrics, 0.1)
trainer.fit(100, 32)