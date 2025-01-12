from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)

# Flatten input dimentions to 1D
X_train = X_train.reshape(60000, 784) # (28*28, width * height)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalise the image pixel values
X_train /= 255
X_test /= 255

# One-Hot Encode the categorical column
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# Build a model architecture with Dense layers
model = Sequential()
model.add(Dense(100, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train the model and make predictions
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))