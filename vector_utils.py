import keras
from numpy import ndarray


class VectorUtils:
    @staticmethod
    def to_column_vector(matrix: ndarray, column_length: int) -> ndarray:
        return matrix.reshape(matrix.shape[0], column_length)

    @staticmethod
    def to_categorical_matrix(column_vector: ndarray, num_rows: int) -> ndarray:
        return keras.utils.to_categorical(column_vector, num_rows)
