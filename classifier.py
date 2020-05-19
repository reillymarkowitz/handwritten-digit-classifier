import keras
from numpy import ndarray
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

class DigitClassifier:
    def __init__(self):
        self.image_size = 28 * 28
        self.num_digits = 10

        self.load_and_shape_data()


    def load_and_shape_data(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.shape_data()


    def shape_data(self) -> None:
        self.x_train = self.to_column_vector(self.x_train)
        self.x_test = self.to_column_vector(self.x_test)

        self.y_train = self.to_categorical(self.y_train)
        self.y_test = self.to_categorical(self.y_test)


    def to_column_vector(self, matrix: ndarray) -> ndarray:
        return matrix.reshape(matrix.shape[0], self.image_size)


    def to_categorical(self, columnVector: ndarray) -> ndarray:
        return keras.utils.to_categorical(columnVector, self.num_digits)


    def initialize_model(self) -> None:
        self.model = Sequential()

        self.model.add(Dense(units=32, activation='sigmoid', input_shape=(self.image_size,)))
        self.model.add(Dense(units=32, activation='sigmoid'))
        self.model.add(Dense(units=self.num_digits, activation='softmax'))


classifier = DigitClassifier()

classifier.initialize_model()

