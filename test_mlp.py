# %% Imports
from pathlib import Path

import numpy as np

import data
from mlp import MLP

# %% Main params
N_SAMPLES = 500

N_FEATURES = 2
HIDDEN_UNITS = [20]

RUNS = 10
EPOCHS = 100
LEARNING_RATE = 0.01

AUTO_DIFF = False
RUN_EAGERLY = False



# %%


def save_history_data(histories, save_path: Path):
    loss = np.array([x.history["loss"] for x in histories])
    val_loss = np.array([x.history["val_loss"] for x in histories])
    bin_acc = np.array([x.history["binary_accuracy"] for x in histories])
    val_bin_acc = np.array([x.history["val_binary_accuracy"]
                           for x in histories])

    epochs = np.array(histories[0].epoch)

    np.save(save_path / "loss", loss)
    np.save(save_path / "val_loss", val_loss)
    np.save(save_path / "bin_acc", bin_acc)
    np.save(save_path / "val_bin_acc", val_bin_acc)
    np.save(save_path / "epochs", epochs)


# %% Create the data
save_path = data.DEFAULT_PATH
u, y = data.create_dataset(N_SAMPLES, save_path)

# %% Plot the data
save_path = data.DEFAULT_PATH
data.plot_data(save_path)

# %% Load data
save_path = data.DEFAULT_PATH
u, y = data.load_data(save_path)

# %% Empty data
histories = []
y_test_true_s = []
y_test_pred_s = []

# %%
for _ in range(RUNS):
    split_i = int(.8*len(y))
    u_train = u[:split_i]
    y_train = y[:split_i]
    u_test = u[split_i:]
    y_test = y[split_i:]

    mlp = MLP([N_FEATURES, *HIDDEN_UNITS, 1])
    mlp.compile(automatic_differentiation=AUTO_DIFF,
                learning_rate=LEARNING_RATE, run_eagerly=RUN_EAGERLY)

    hist = mlp.fit(u_train, y_train, batch_size=1,
                   epochs=EPOCHS, validation_split=0.2)
    histories.append(hist)
    y_test_pred = mlp.predict_class(u_test)

    y_test_true_s.append(y_test)
    y_test_pred_s.append(y_test_pred)


# %% Save
np.save(save_path / 'test_true', np.array(y_test_true_s))
np.save(save_path / 'test_pred', np.array(y_test_pred_s))

save_history_data(histories, save_path)
