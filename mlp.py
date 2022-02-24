import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.datasets
import matplotlib.pyplot as plt

N_FEATURES = 2
HIDDEN_UNITS = [5]
N = 500


RUN_EAGERLY = True

# MLP uses simple forward model so Sequential is subclassed


class MLP(keras.Sequential):
    def __init__(self, units_config):
        super().__init__()
        self.n_dense_layers = len(units_config) - 1

        # Input shape is known, so provided (excludes batch_size)
        self.add(keras.layers.Input((units_config[0],)))

        # A dense layer is added for the listed number units in each layer.
        for n in units_config[1:]:
            # NOTE: Sigmoid for some reason does not work, not sure yet why
            self.add(keras.layers.Dense(n, activation='relu',
                                        kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        self.add(keras.layers.Dense(units_config[-1], activation=None,
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        # Sigmoid is applied to output layer to produce probability distribution (binary classification)

        self.summary()

    # def call(self, inputs, training=False, mask=None):
    #     u = inputs
    #     for layer in range(self.n_layers):
    #         x = self.layers[layer](u)
    #         a =

    #     return outputs
    # def calc_gradients(self, deltas, theta, X):
    #     # For each hidden layer
    #     for layer in range(self.n_hidden_layers - 1, -1, -1):
    #         # Find length of current layer
    #         l_m0 = tf.shape(theta[2*layer])[0]
    #         delta_m = [0] * l_m0
    #         # For every hidden unit in the layer
    #         for i in range(l_m0):
    #             # Compute sigmoid derivative with activation of i-th unit in current layer
    #             d_sig = X[layer][0, i] * (1 - X[layer][0, i])
    #             # Get length of previous layer
    #             l_m1 = tf.shape(theta[2*(layer - 1)])[0]
    #             delta_m_i = 0
    #             for j in range(l_m1):
    #                 delta_m_i += deltas[-1][0, j] * theta[2*layer][i, j]
    #             delta_m_i *= d_sig
    #             delta_m[i] = delta_m_i

    # Implementation of the train step function for a batch of data.
    def train_step(self, data):
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

        def d_dx_MSE(x, y):
            return 2 * (x - y)

        # Compute deltas for last layer
        d_dx_mse = d_dx_MSE(X[-1], y)
        delta_k = d_dx_mse * X[-1] * (1 - X[-1])
        deltas.append(delta_k)

        # Compute deltas for hidden layers
        for layer in range(self.n_dense_layers - 1, 0, -1):
            # Makes W_k [batch_size x L^k x L^(k+1)]
            W_k = tf.expand_dims(theta[2 * layer], axis=0)
            W_k_tile = tf.tile(W_k, [batch_size, 1, 1])

            # [batch size x L^k]
            d_sigma = X[layer] * (1 - X[layer])
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

    def compile(self, automatic_differentiation=False, learning_rate=0.01, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, **kwargs):
        self.automatic_differentiation = automatic_differentiation
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9) # Without momentum the optimizer is hopelessly bad
        # Only MSE is supported due to hard coded back propagation on MSE loss. TODO
        # loss = keras.losses.BinaryCrossentropy(from_logits=True) NOTE: Binary crossentropy might be preferred due to binary class prediction.
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.BinaryAccuracy()]
        return super().compile(optimizer, loss, metrics, loss_weights,
                               weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def predict_class(self, x, **kwargs):
        y_pred = self.predict(x, **kwargs)
        return (y_pred > 0.5).reshape((-1,)).astype(np.int32)


def load_data(N):
    gq = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.7, n_samples=N, n_features=2,
        n_classes=2, shuffle=True, random_state=None)
    return gq


def plot_data(X, Y):
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
    plt.show()


def main():
    u, y = load_data(N)
    plot_data(u, y)

    split_i = int(.8*N)
    u_train = u[:split_i]
    y_train = y[:split_i]
    u_test = u[split_i:]
    y_test = y[split_i:]

    mlp = MLP([N_FEATURES, *HIDDEN_UNITS, 1])
    mlp.compile(automatic_differentiation=True, learning_rate=0.02, run_eagerly=RUN_EAGERLY)

    mlp.fit(u_train, y_train, batch_size=1, epochs=100, validation_split=0.2)
    y_test_pred = mlp.predict_class(u_test)

    print(np.mean(y_test == y_test_pred, axis=0))
    pass


if __name__ == "__main__":
    main()
