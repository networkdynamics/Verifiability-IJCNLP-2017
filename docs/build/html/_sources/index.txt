.. verifiability documentation master file, created by
   sphinx-quickstart on Sun Feb 26 17:43:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quote verifiability, part of the Validators Project
===================================================

This documentation isn't displaying correctly on Gitlab.  Read it 
`here <http://cgi.cs.mcgill.ca/~enewel3/temp/verifiability/docs/html/>`_
instead.

The ``HarmonicLogistic`` model
------------------------------

The ``HarmonicLogistic`` class is a machine-learning model that produces a
regression in the range [0,1].  It is structured as a harmonic mean of logistic
regressions.  The model can have any number of logistic regression units, each
being a full standard logistic regression model.

The generative story of the model is that some phenomenon, like quote
verifiability, results from multiple *factors* simultaneously holding---in the
case of verifiability, these are the component verifiability of the source,
cue, and content spans.  The overall verifiability score is a result of
combining the factors with the idea that the *weakest link* is the main
determinant of the overall verifiability.  Conceptually, this could be achieved
by taking the ``min`` of the *factor*\ |s| verifiabilities, but this creates
discontinuities in the derivative of the model's output.  Instead, the harmonic
mean is used as a kind of *soft* ``min``, which is differentiable.

Using the ``HarmonicLogistic`` model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Like many machine learning models, the ``HarmonicLogistic`` model expects
each example to be encoded as a numeric feature vector.  One crucial detail is
that you need to tell the ``HarmonicLogistic`` model which features belong to
which *factor*.  So, for example, let us suppose that we have a 10-component
feature vector, where features 0 through 2 describe the source, features 3
through 6 describe the cue, and features 7 through 9 describe the
content.  The source, cue, and content represent the three *factors* that we
would like to model using individual logistic regressions, and we need to tell
``HarmonicLogistic`` which features belong to which factors.

We do this using the ``lengths`` parameter when creating an instance:

.. code-block:: python

    from harmonic_logistic import HarmonicLogistic

    regressor = HarmonicLogistic(lengths=[3, 4, 3])

Training
^^^^^^^^
To train a ``HarmonicLogistic`` model, you need to encode the training data
into two numpy arrays: one for the feature vectors, and one for the true output
scores (the value we're trying to regress, i.e. the verifiability score).

The feature vectors should be stored in a 2-dimensional numpy array, such that
each row of the array is one feature vector, that is, the shape of the array
should be ``(num_training_examples, feature_vector_length)``.  The target
scores (verifiability scores to be regressed) should be in a 1-dimensional
array, with length equal to the number of training examples.  Both arrays
should have ``dtype='float64'``.

Training the model might look something like this:

.. code-block:: python

    import numpy as np

    # Get the training data
    feature_vectors = [
        [1, 2, -0.2, 3, 2, -1, 0, 10, 0.1, -1],
        [2, 1, -0.4, 3, 0, -2, 0, 10, 0.2, 11],
        [3, 0, 0.54, 1, 0, -3, 1, 11, 0.7, -4],
    ]
    target_vector = [0.4, 0.7, 0.8]

    # Cast it into numpy float64 arrays
    feature_vectors = np.array(feature_vectors, dtype='float64')
    target_vector = np.array(target_vector, dtype='float64')
    
    # Train the model
    regressor.fit(feature_vectors, target_vector)

By default, the model will train until it's loss function changes by less than
``1e-8`` from one stochastic-gradient-descent step to the next, and will use a
learning rate of ``1.0``.  The loss and the change in loss are printed after
each training epoch.  You can control all of these behaviors, e.g.:

.. code-block:: python

    # Specify the learning rate when constructing the model
    regressor = HarmonicLogistic(lengths=[3, 4, 3], learning_rate=0.1)

    # Train the model
    regressor.fit(
        feature_vectors, target_vector, 
        tolerance=1e-10, learning_rate=1.0, verbose=False
    )

How do you know when to modify the learning rate?  Setting the learning rate
involves weighing the speed of training against the stability of the stochastic
gradient descent and the precision of the fit it is capable of generating.  The
default learning rate of 1.0 worked well in the test suite, and if it works and
the model doesn't take too long to train, then it's fine.

A larger step size means that the model will approach convergence more quickly,
at least at first.  But if the step size is too large, then the model may never
converge, because it actually steps over the basin of the loss-function.  A
smaller step size will take longer to converge, but because of the smaller step
size, the algorithm can home in on a minimum of the loss function more
precisely.

What should the tolerance be?  The tolerance is involves a similar time vs
accuracy tradeoff.  It represents an amount of change in the model's loss
function that we consider to be negligible.  Once the loss function changes by
less than the tolerance from one sgd-step to the next, optimization will stop. 
A very small (i.e. very precise) tolerance might require a smaller learning
rate to converge.

Predicting
^^^^^^^^^^
A model can be used to predict the output score (verifiability) from supplied
feature vectors.  Supply the feature vectors to the ``predict`` method in the
same format as for the `train` method, but don't provide a target vector:

.. code-block:: python

    >>> regressor.predict(feature_vectors)
    array([0.349945699, 0.70010234, 0.77732115])

Saving and loading
^^^^^^^^^^^^^^^^^^
Once trained, save a model to disk using the ``save()`` method, passing it a path at
which to write the model.  The internal parameters that define the model will
be written to file using ``numpy``\ |s| ``.npz`` format.  To load a model,
supply a model file's path to the ``load`` keyword in the constructor, or 
call the ``load()`` method on an existing model instance:

.. code-block:: python

    # Save a model
    regressor.save('my-model.npz')

    # Load a model using the load keyword in the constructor
    new_regressor = HarmonicLogistic(load='my-model.npz')

    # Or load a model onto an existing HarmonicLogistic instance
    new_regressor = HarmonicLogistic(lengths=[3,4,3])   # Note: lengths overwritten by those in the stored model
    new_regressor.load('my-model.npz')




.. |s| replace:: |rsquo|\ s
.. |rsquo| unicode:: 0x2019 .. right single quote
