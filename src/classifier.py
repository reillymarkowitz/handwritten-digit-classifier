from keras.callbacks import History
from numpy import ndarray
from model import Model


class DigitClassifier:
    def __init__(self):
        self.image_size = 28 * 28
        self.num_digits = 10

        self.model = Model(input_size=self.image_size, output_size=self.num_digits)

    def train_model(self, x_train: ndarray, y_train: ndarray) -> History:
        return self.model.train(x_train, y_train)

    def evaluate_model(self, x_test: ndarray, y_test: ndarray) -> None:
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print(f'Test loss: {test_loss:.3}')
        print(f'Test accuracy: {test_accuracy:.3}')
