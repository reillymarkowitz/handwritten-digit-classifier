from keras.datasets import mnist
from keras.callbacks import History
from model import Model
from plotter import Plotter
from vector_utils import VectorUtils


class DigitClassifier:
    def __init__(self):
        self.image_size = 28 * 28
        self.num_digits = 10

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.model = Model(input_size=self.image_size, output_size=self.num_digits)

        self.shape_data()

    def shape_data(self) -> None:
        self.x_train = VectorUtils.to_column_vector(matrix=self.x_train, column_length=self.image_size)
        self.x_test = VectorUtils.to_column_vector(matrix=self.x_test, column_length=self.image_size)

        self.y_train = VectorUtils.to_categorical_matrix(column_vector=self.y_train, num_rows=self.num_digits)
        self.y_test = VectorUtils.to_categorical_matrix(column_vector=self.y_test, num_rows=self.num_digits)

    def compile_model(self) -> None:
        self.model.compile()

    def train_model(self) -> History:
        return self.model.train(self.x_train, self.y_train)

    def evaluate_model(self) -> None:
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f'Test loss: {test_loss:.3}')
        print(f'Test accuracy: {test_accuracy:.3}')


classifier = DigitClassifier()
classifier.compile_model()

training_result = classifier.train_model()
history = training_result.history
Plotter.plot(history['accuracy'], history['val_accuracy'])

classifier.evaluate_model()
