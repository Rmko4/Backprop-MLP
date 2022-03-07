from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MLP(keras.Sequential):
    # MLP uses simple forward model so Sequential is subclassed

    def __init__(self, units_config: List[int]):
        super().__init__()
        self.n_dense_layers = len(units_config) - 1

        # Input shape is known, so provided (excludes batch_size)
        self.add(keras.layers.Input((units_config[0],)))

        # A dense layer is added for the listed number units in each layer.
        for n in units_config[1:-1]:
            keras.activations.relu
            self.add(keras.layers.Dense(n, activation='relu',
                                        kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        self.add(keras.layers.Dense(units_config[-1], activation='sigmoid',
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        # Sigmoid is applied to output layer to produce probability distribution (binary classification)

        self.summary()

    def train_step(self, data):
        # Implementation of the train step function for a batch of data.
        if self.automatic_differentiation:
            return super().train_step(data)
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        u, y = data
        y = tf.cast(y, tf.float32)
        y = tf.expand_dims(y, axis=-1)

        # The batch size will be the outer dimension (dynamically checked)
        batch_size = tf.shape(u)[0]

        X = []  # Activations of units in each layer k
        deltas = []  # Respective deltas for layer k + 1

        # Forward pass
        X.append(u)
        x = u
        for layer in range(self.n_dense_layers):
            x = self.layers[layer](x)
            X.append(x)

        theta = self.trainable_variables

        def d_dypred_binary_crossentropy(y_true, y_pred):
            eps = keras.backend.epsilon()
            return (1 - y_true) / (1 - y_pred + eps) - (y_true / y_pred + eps)

        def d_dx_sigmoid(x):
            return x * (1 - x)

        def d_dx_ReLU(x):
            return 1. * tf.cast((x > 0), tf.float32)

        # Compute deltas for last layer
        d_dx_loss = d_dypred_binary_crossentropy(y, X[-1])
        delta_k = d_dx_loss * d_dx_sigmoid(X[-1])
        deltas.append(delta_k)

        # Compute deltas for hidden layers
        for layer in range(self.n_dense_layers - 1, 0, -1):
            # Makes W_k [batch_size x L^k x L^(k+1)]
            W_k = tf.expand_dims(theta[2 * layer], axis=0)
            W_k_tile = tf.tile(W_k, [batch_size, 1, 1])

            # [batch size x L^k]
            d_sigma = d_dx_ReLU(X[layer])
            dak1_dak = tf.linalg.matvec(W_k_tile, deltas[-1])
            delta_k = d_sigma * dak1_dak

            deltas.append(delta_k)

        theta_gradient = []
        for layer in range(len(deltas)):
            # Outer product of the activations in layer k and deltas in layer k - 1,
            # yields the gradient of weight for layer k
            W_gradient = tf.einsum(
                'ni,nj->nij', X[layer], deltas[-(layer + 1)])
            # Reduces to mean over batch axis to get mean gradient for batch
            W_gradient = tf.reduce_mean(W_gradient, axis=0)
            theta_gradient.append(W_gradient)

            # Bias gradient are given by the deltas from the corresponding layer
            bias_gradient = deltas[-(layer + 1)]
            bias_gradient = tf.reduce_mean(bias_gradient, axis=0)
            theta_gradient.append(bias_gradient)

        # Update weights by providing the optimizer (SGD) with the gradient.
        self.optimizer.apply_gradients(zip(theta_gradient, theta))

        # Update metrics (includes the metric that tracks the loss)
        y_hat = X[-1]
        self.compiled_metrics.update_state(y, y_hat)
        loss_value = self.compiled_loss(y, y_hat)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compile(self, automatic_differentiation=False,
                learning_rate=0.02, run_eagerly=None, **kwargs):
        self.automatic_differentiation = automatic_differentiation
        # Stochastic Gradient Descent
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        # Binary Cross Entropy
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = [keras.metrics.BinaryAccuracy()]
        run_eagerly = None if run_eagerly == False else run_eagerly
        return super().compile(optimizer, loss, metrics, run_eagerly, **kwargs)

    def predict_class(self, x, **kwargs):
        y_pred = self.predict(x, **kwargs)
        return (y_pred > 0.5).reshape((-1,)).astype(np.int32)
