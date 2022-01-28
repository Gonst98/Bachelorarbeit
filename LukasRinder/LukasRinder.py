import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

def plot_heatmap_2d(dist, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, mesh_count=1000, name=None):
    plt.figure()
    
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    
    concatenated_mesh_coordinates = tf.transpose(tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
    prob = dist.prob(concatenated_mesh_coordinates)
    #plt.hexbin(concatenated_mesh_coordinates[:,0], concatenated_mesh_coordinates[:,1], C=prob, cmap='rainbow')
    prob = prob.numpy()
    
    plt.imshow(tf.transpose(tf.reshape(prob, (mesh_count, mesh_count))), origin="lower")
    plt.xticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [xmin, xmin/2, 0, xmax/2, xmax])
    plt.yticks([0, mesh_count * 0.25, mesh_count * 0.5, mesh_count * 0.75, mesh_count], [ymin, ymin/2, 0, ymax/2, ymax])
    if name:
        plt.savefig(name + ".png", format="png")
        
def load_and_preprocess_mnist(logit_space=True, batch_size=128, shuffle=True, classes=-1, channels=False):
    """
     Loads and preprocesses the MNIST dataset. Train set: 50000, val set: 10000,
     test set: 10000.
    :param logit_space: If True, the data is converted to logit space.
    :param batch_size: batch size
    :param shuffle: bool. If True, dataset will be shuffled.
    :param classes: int of class to take, defaults to -1 = ALL
    :return: Three batched TensorFlow datasets:
    batched_train_data, batched_val_data, batched_test_data.
    """

    (x_train, y_train), (x_test, y_test) = tfkd.mnist.load_data()

    # reserve last 10000 training samples as validation set
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # if logit_space: convert to logit space, else: scale to [0, 1]
    if logit_space:
        x_train = logit(tf.cast(x_train, tf.float32))
        x_test = logit(tf.cast(x_test, tf.float32))
        x_val = logit(tf.cast(x_val, tf.float32))
        interval = 256
    else:
        x_train = tf.cast(x_train / 256, tf.float32)
        x_test = tf.cast(x_test / 256, tf.float32)
        x_val = tf.cast(x_val / 256, tf.float32)
        interval = 1


    if classes == -1:
        pass
    else:
        #TODO: Extract Multiple classes: How to to the train,val split,
        # Do we need to to a class balance???
        x_train = np.take(x_train, tf.where(y_train == classes), axis=0)
        x_val = np.take(x_val, tf.where(y_val == classes), axis=0)
        x_test = np.take(x_test, tf.where(y_test == classes), axis=0)

    # reshape if necessary
    if channels:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28, 1))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28, 1))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28, 1))
    else:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28))

    if shuffle:
        shuffled_train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)

    batched_train_data = shuffled_train_data.batch(batch_size)
    batched_val_data = tf.data.Dataset.from_tensor_slices(x_val).batch(batch_size)
    batched_test_data = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)    
    
    return batched_train_data, batched_val_data, batched_test_data, interval

@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))

@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss

class Made(tf.keras.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.
    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)