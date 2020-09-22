# Bayesian Confidence Propagation Neural Network (BCPNN)

> "A Bayesian Confidence Propagation Neural Network (BCPNN) is an artificial neural network inspired by Bayes' theorem: node activations represent probability ("confidence") in the presence of input features or categories, synaptic weights are based on estimated correlations and the spread of activation corresponds to calculating posteriori probabilities. It was originally proposed by Anders Lansner and Örjan Ekeberg at KTH." - [Wikipedia](https://en.wikipedia.org/wiki/Bcpnn)

Implemented roughly following the architecture devised in  "The Use of a Bayesian Neural Network Model for Classification Tasks", Anders Holst, 1997 [[1]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.4318&rep=rep1&type=pdf).

The single-layered feedforward version can be used for classification tasks, while the recurrent version works as a [CAM](https://en.wikipedia.org/wiki/Content-addressable_memory).

## Usage
Instantiate classifier, then `fit`, and `predict` or `predict_proba` for classification. The API closely follows the method naming convention of sklearn's classifiers, and so should be straightforward to use for familiar users.

## Dependencies
Mainly numpy + sklearn for its OneHotEncoder, and pytest for running tests. Run `$ pip install -r requirements(_dev).txt` to install.

![Denoising demo](denoising_demo.png?raw=true)
