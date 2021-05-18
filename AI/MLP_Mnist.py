import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split



def mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0]), -1)
    x_test = x_test.reshape((x_test.shape[0]), -1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # build model
    model = Sequential([
        Dense(784, input_shape=(784,)),
        Activation('relu'),
        Dropout(0.5),
        Dense(100),
        Activation('relu'),
        Dropout(0.5),
        Dense(50),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ])

    Adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = Adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = model.fit(x_train, y_train, batch_size = 500, validation_split = 0.3, epochs = 100)

    for i in range(100):
        acc = hist.history['accuracy']
        loss = hist.history['loss']
        print('epoch = ', i, ', loss: ', loss[i], ', accuracy: ', acc[i])
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('accuracy: ', test_acc)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training accuracy', 'test accuracy'], loc = 'lower right')
    plt.show()

    y_pred = model.predict(x_test)
    idx = np.random.randint(0,10000)
    title = 'Label is : ' + str(np.argmax(y_test[idx])) + ' and Predicted : ' + str(np.argmax(y_pred[idx]))

    img = x_test[idx].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()



if __name__ == '__main__':
    print('start')
    mnist()
