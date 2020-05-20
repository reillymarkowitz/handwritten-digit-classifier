import keras
import matplotlib.pyplot as plt
from numpy import ndarray
from model import Model
from keras.datasets import mnist
from keras.callbacks import History


class DigitClassifier:
    def __init__(self):
        self.image_size = 28 * 28
        self.num_digits = 10

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.model = Model(input_size=self.image_size, output_size=self.num_digits)

        self.shape_data()

    def shape_data(self) -> None:
        self.x_train = self.to_column_vector(self.x_train)
        self.x_test = self.to_column_vector(self.x_test)

        self.y_train = self.to_categorical(self.y_train)
        self.y_test = self.to_categorical(self.y_test)

    def to_column_vector(self, matrix: ndarray) -> ndarray:
        return matrix.reshape(matrix.shape[0], self.image_size)

    def to_categorical(self, column_vector: ndarray) -> ndarray:
        return keras.utils.to_categorical(column_vector, self.num_digits)

    def train_model(self) -> History:
        return self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=50, validation_split=.1)

    def evaluate_model(self) -> tuple:
        return self.model.evaluate(self.x_test, self.y_test)


classifier = DigitClassifier()

classifier.model.compile()

training_result = classifier.train_model()

test_loss, test_accuracy = classifier.evaluate_model()
print(f'Test loss: {test_loss:.3}')
print(f'Test accuracy: {test_accuracy:.3}')
