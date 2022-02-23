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
            self.add(keras.layers.Dense(n, activation=None,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
            # can apply sigmoid to output layer for binary classification
            self.add(keras.layers.Activation('sigmoid'))
        self.summary()

    # def call(self, inputs, training=False, mask=None):
    #     u = inputs
    #     for layer in range(self.n_layers):
    #         x = self.layers[layer](u)
    #         a =

    #     return outputs

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        u, y = data
        y = tf.cast(y, tf.float32)
        y = tf.expand_dims(y, axis=-1)

        X = []
        A = []
        delta = []
        # Forward pass
        X.append(u)
        x = u
        # a denotes the potential the i-th layer
        # x denotes the activation of the i-th layer
        for layer in range(self.n_hidden_layers):
            a = self.layers[2*layer](x)
            x = self.layers[2*layer + 1](a)
            A.append(a)  # TODO: Get rid of later
            X.append(x)

        theta = self.trainable_variables
        # Compute delta for last layer
        d_dx_mse = d_dx_MSE(X[-1], y)
        delta.append(d_dx_mse * X[-1] * (1 - X[-1]))

        # Compute deltas for hidden layers
        for layer in range(self.n_hidden_layers - 1, -1, -1):
            d_sigma = X[layer] * (1 - X[layer])
            new_delta = d_sigma * theta[2 * layer] * delta[-1]
            delta.append(new_delta)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
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
    mlp.fit(u, y, 10, 100, validation_split=0.2)
    pass


if __name__ == "__main__":
    main()
