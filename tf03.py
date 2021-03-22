import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.38:
            print("Loss is too low cancelling it!")
            self.model.stop_training = True


callbacks = MyCallback()
(x_train, y_train),  (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

print(x_test.shape)
x_train = x_train.reshape(-1, 28, 28, 1)/255
x_test = x_test.reshape(-1, 28, 28, 1)/255
print(x_test.shape)
print(x_test[0].shape)


model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=6, callbacks=[callbacks])
model.evaluate(x_test, y_test)


# layer_outputs = [layer.output for layer in model.layers]
#
# activation_model = keras.models.Model(inputs=model.input, output=layer_outputs)

print(x_test[0].reshape(-1, 28, 28, 1).shape)
predict = model.predict([x_test[10].reshape(-1, 28, 28, 1)])
test_idx = np.argmax(predict)
print(y_test[10], ">>>", test_idx)
print(predict[0].shape)
# plt.imshow(predict[0][0, :, :, 1])
# plt.show()
