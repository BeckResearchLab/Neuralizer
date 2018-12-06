Neuralizer
==========

![Neuralizer logo](https://github.com/BeckResearchLab/Neuralizer/blob/organizer/Logo.jpg)

Neuralizer, is a tool to find substitute neural network model for complicated physics based, kinetic & flux models. Examples include: P2D model of Li-ion battery [Murbach2017](#Murbach2017). Models of these types are typically composed of a complex set of PDE or ODEs. This makes solving these models computationally complex and expensive.  Moreover, using these models that take seconds to solve prohibits them from being used in real-time situations like process control that require subsecond solutions. In contrast, neural network models are less time-consuming and computionally expensive to use for producing solutions and when properly trained, can provide nearly as accurate predictions as the complete physics, kinetic or flux based model.

The module is built on top of Keras and Tensorflow.  It also depends on the other usual suspects of the scientific Python stack such as pandas, scikit-learn and numpy.

Use cases
----------------
* Preprocess the sensitivity analysis results of a physics based, kinetic or flux model
* Find a set of neural network hyperparameters that yield a high quality surrogate model
* For a given surrogate model, evaluate the model against physics based, kinetic or flux model
* Run the neural network surrogate model

Getting started
-----------------
* Install conda
* Create a virtual environment using conda and the `environment.yml` file supplied in this repository
* Use the virtual environment to proprocess the input with [data_process.py](https://github.com/BeckResearchLab/Neuralizer/blob/organizer/model_create/data_process.py)
* ...

References
-----------------
<a name="Murbach2017">Murbach2017: http://paper.location "P2D model stuff"</a>
