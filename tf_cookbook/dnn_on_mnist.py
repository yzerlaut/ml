import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def visualize_mnist():
    # reload from scratch
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    import sys
    sys.path.append('../..')
    import matplotlib.pylab as plt
    from graphs.my_graph import graphs
    mg = graphs()

    nx, ny = 8, 4
    fig, AX = mg.figure(axes=(ny,nx), hspace=0., wspace=0.)
    for i in range(ny*nx):
        AX[int(i/nx)][i%nx].imshow(X_train[i], cmap=plt.cm.binary)
        AX[int(i/nx)][i%nx].axis('off')
    mg.show()


print(X_train.shape)    
# feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
# dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
#                                      feature_columns=feature_cols)

# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
# dnn_clf.train(input_fn=input_fn)

    
