 Neuralizer

![Neuralizer logo](https://github.com/BeckResearchLab/Neuralizer/blob/master/Logo.png)

Neuralizer, is a tool to find substitute neural network model for complicated physics based, kinetic & flux models. Examples include: P2D model of Li-ion battery [Murbach2017](#Murbach2017). Models of these types are typically composed of a complex set of PDE or ODEs. This makes solving these models computationally complex and expensive.  Moreover, using these models that take seconds to solve prohibits them from being used in real-time situations like process control that require subsecond solutions. In contrast, neural network models are less time-consuming and computionally expensive to use for producing solutions and when properly trained, can provide nearly as accurate predictions as the complete physics, kinetic or flux based model.

The module is built on top of Keras and Tensorflow.  It also depends on the other usual suspects of the scientific Python stack such as pandas, scikit-learn and numpy.

Organization of project 
-----------------------
    Neuralizer/
      |- README.md
      |- neuralizer/
         |- __init__.py
         |- data_process.py
         |- model_create.py
         |- param_record.py
         |- parameter.py
         |- latest.json
         |- data/
            |- ...
         |- tests/
            |- ...
         |- Results
            |- ...
      |doc/

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
* Do coarse and broader  search first with [parameter.py](https://github.com/BeckResearchLab/Neuralizer/blob/organizer/model_create/parameter.py)
* After Getting a bucket of better paramter combinations, do complete search with [complete_search.py](https://github.com/BeckResearchLab/Neuralizer/blob/organizer/model_create/complete_search.py)
* The intermediate, latest and best results along with their configuration are stored in [Results](https://github.com/BeckResearchLab/Neuralizer/tree/organizer/model_create/Results)


References
-----------------
<a name="Murbach2017">Murbach2017: http://paper.location "P2D model stuff"</a>
