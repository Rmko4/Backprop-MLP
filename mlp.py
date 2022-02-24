import tensorflow as tf
import tensorflow.keras as keras
import sklearn.datasets

# def d_dx_sigmoid(x):
#     return tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))


def d_dx_MSE(x, y):
    return 2 * (x - y)


class MLP(keras.Sequential):
    def __init__(self, n_features, n_units):
        super().__init__()
        self.n_hidden_layers = len(n_units)

        self.add(keras.layers.Input((n_features,)))

        for n in n_units:
            self.add(keras.layers.Dense(n, activation='sigmoid',
                                        kernel_initializer='glorot_uniform', bias_initializer='zeros'))
            # can apply sigmoid to output layer for binary classification
            # TODO: You don't need this split as we are only interested in the activations due to:  d/dx sigmoid ~ x^k.
        self.summary()

    # def call(self, inputs, training=False, mask=None):
    #     u = inputs
    #     for layer in range(self.n_layers):
    #         x = self.layers[layer](u)
    #         a =

    #     return outputs
    def calc_gradients(self, deltas, theta, X):
        # For each hidden layer
        for layer in range(self.n_hidden_layers - 1, -1, -1):
            # Find length of current layer
            l_m0 = tf.shape(theta[2*layer])[0]
            delta_m = [0] * l_m0
            # For every hidden unit in the layer
            for i in range(l_m0):
                # Compute sigmoid derivative with activation of i-th unit in current layer
                d_sig = X[layer][0, i] * (1 - X[layer][0, i])
                # Get length of previous layer
                l_m1 = tf.shape(theta[2*(layer - 1)])[0]
                delta_m_i = 0
                for j in range(l_m1):
                    delta_m_i += deltas[-1][0, j] * theta[2*layer][i, j]
                delta_m_i *= d_sig
                delta_m[i] = delta_m_i

    # Does still support mini_batch
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        u, y = data
        y = tf.cast(y, tf.float32)
        y = tf.expand_dims(y, axis=-1)

        # The batch size will be the outer dimension (dynamically checked)
        batch_size = tf.shape(u)[0]

        X = []
        deltas = []

        # Forward pass
        X.append(u)
        x = u

        # a denotes the potential the i-th layer
        # x denotes the activation of the i-th layer
        for layer in range(self.n_hidden_layers):
            x = self.layers[layer](x)
            X.append(x)

        theta = self.trainable_variables

        # Compute deltas for last layer
        d_dx_mse = d_dx_MSE(X[-1], y)
        delta_k = d_dx_mse * X[-1] * (1 - X[-1])
        deltas.append(delta_k)

        # self.calc_gradients(deltas, theta, X)

        # Compute deltas for hidden layers
        for layer in range(self.n_hidden_layers - 1, 0, -1):
            # Makes W_k [batch_size x L^k x L^(k+1)]
            W_k = tf.expand_dims(theta[2 * layer], axis=0)
            W_k_tile = tf.tile(W_k, [batch_size, 1, 1])

            # W_k is [L^k x L^(k+1)]
            # NOTE: This weight matrix is transpose of usual notation
            # W_k = theta[2 * layer]

            # [batch size * L^k]
            d_sigma = X[layer] * (1 - X[layer]) 
            dak1_dak = tf.linalg.matvec(W_k_tile, deltas[-1])
            delta_k = d_sigma * dak1_dak

            deltas.append(delta_k)

        theta_gradient = []
        for layer in range(len(deltas)):
            W_gradient = tf.einsum('ni,nj->nij', X[layer], deltas[-(layer + 1)])
            W_gradient = tf.reduce_sum(W_gradient, axis=0)
            theta_gradient.append(W_gradient)
            bias_gradient = deltas[-(layer + 1)]
            bias_gradient = tf.reduce_sum(bias_gradient, axis=0)
            theta_gradient.append(bias_gradient)

        # Update weights
        self.optimizer.apply_gradients(zip(theta_gradient, theta))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, X[-1])
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compile(self, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, **kwargs):
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
        loss = 'mse'
        return super().compile(optimizer, loss, metrics, loss_weights,
                               weighted_metrics, run_eagerly, steps_per_execution, **kwargs)


def load_data():
    N = 500
    gq = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.7, n_samples=N, n_features=2,
        n_classes=2, shuffle=True, random_state=None)
    return gq


def main():
    mlp = MLP(2, [100, 1])
    mlp.compile(run_eagerly=True)
    u, y = load_data()
    mlp.fit(u, y, 1, 5, validation_split=0.2)
    pass


if __name__ == "__main__":
    main()
