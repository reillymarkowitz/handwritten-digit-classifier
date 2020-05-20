from typing import Tuple
from keras.datasets import mnist
from numpy import ndarray
from src.vector_utils import VectorUtils


class DataProvider:
    def __init__(self, image_size: int, num_digits: int):
        self.image_size = image_size
        self.num_digits = num_digits

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.format_training_data()
        self.format_test_data()

    def format_training_data(self):
        self.x_train = self.to_column_vector(matrix=self.x_train)
        self.y_train = self.to_categorical_matrix(column_vector=self.y_train)

    def format_test_data(self):
        self.x_test = self.to_column_vector(matrix=self.x_test)
        self.y_test = self.to_categorical_matrix(column_vector=self.y_test)

    def to_column_vector(self, matrix: ndarray) -> ndarray:
        return VectorUtils.to_column_vector(matrix, column_length=self.image_size)

    def to_categorical_matrix(self, column_vector: ndarray) -> ndarray:
        return VectorUtils.to_categorical_matrix(column_vector, num_rows=self.num_digits)

    def get_training_data(self) -> Tuple[ndarray, ndarray]:
        return self.x_train, self.y_train

    def get_test_data(self) -> Tuple[ndarray, ndarray]:
        return self.x_test, self.y_test
