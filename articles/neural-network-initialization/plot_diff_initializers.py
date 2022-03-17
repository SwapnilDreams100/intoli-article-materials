import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import initializers
from keras.datasets import mnist
from keras.utils import np_utils
from utils import (
    compile_model,
    create_mlp_model,
    get_activations,
    grid_axes_it,
)

seed = 10

# Number of points to plot
n_train = 1000
n_test = 100
n_classes = 10

# Network params
n_hidden_layers = 5
dim_layer = 100
batch_size = n_train
epochs = 1

# Load and prepare MNIST dataset.
n_train = 60000
n_test = 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(np.unique(y_test))
data_dim = 28 * 28

x_train = x_train.reshape(60000, 784).astype('float32')[:n_train]
x_test = x_test.reshape(10000, 784).astype('float32')[:n_train]
x_train /= 255
x_test /= 255

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# Run the data through a few MLP models and save the activations from
# each layer into a Pandas DataFrame.
rows = []
init = initializers.GlorotNormal(seed=seed)
#     activation = 'relu'
activation = 'tanh'
# activation = 'sigmoid'

model = create_mlp_model(
    n_hidden_layers,
    dim_layer,
    (data_dim,),
    n_classes,
    init,
    'zeros',
    activation
)
compile_model(model)
output_elts = get_activations(model, x_test)
n_layers = len(model.layers)
i_output_layer = n_layers - 1

for i, out in enumerate(output_elts[:-1]):
    if i > 0 and i != i_output_layer:
        for out_i in out.ravel()[::20]:
            rows.append([i, out_i])

df = pd.DataFrame(rows, columns=['Hidden Layer', 'Output'])

# Plot previously saved activations from the 5 hidden layers
# using different initialization schemes.
fig = plt.figure(figsize=(12, 6))
axes = grid_axes_it(1, 1, fig=fig)

ax = next(axes)
ddf = df
sns.violinplot(x='Hidden Layer', y='Output', data=ddf, ax=ax, scale='count', inner=None)

ax.set_xlabel('')
ax.set_ylabel('')

ax.set_title('Weights Drawn from glorot', fontsize=13)

ax.set_ylabel(activation+ " Neuron Outputs")
ax.set_xlabel("Hidden Layer")

plt.tight_layout()
plt.show()
fig.savefig('glorot.png', dpi=fig.dpi)
