from keras.callbacks import History
from numpy import ndarray
from model import Model
from data_provider import DataProvider
from plotter import Plotter


class DigitClassifier:
    def __init__(self):
        self.image_size = 28 * 28
        self.num_digits = 10

        self.model = Model(input_size=self.image_size, output_size=self.num_digits)

    def compile_model(self) -> None:
        self.model.compile()

    def train_model(self, x_train: ndarray, y_train: ndarray) -> History:
        return self.model.train(x_train, y_train)

    def evaluate_model(self, x_test: ndarray, y_test: ndarray) -> None:
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print(f'Test loss: {test_loss:.3}')
        print(f'Test accuracy: {test_accuracy:.3}')


classifier = DigitClassifier()
classifier.compile_model()

provider = DataProvider(classifier.image_size, classifier.num_digits)

x_train, y_train = provider.get_training_data()
training_result = classifier.train_model(x_train, y_train)

history = training_result.history
Plotter.plot(history['accuracy'], history['val_accuracy'])


x_test, y_test = provider.get_test_data()
classifier.evaluate_model(x_test, y_test)
