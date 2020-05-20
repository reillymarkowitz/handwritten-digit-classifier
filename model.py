from keras.models import Sequential
from keras.layers import Dense


class Model(Sequential):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.construct()

    def construct(self) -> None:
        self.add(Dense(units=32, activation='sigmoid', input_shape=(self.input_size,)))
        self.add(Dense(units=32, activation='sigmoid'))
        self.add(Dense(units=self.output_size, activation='softmax'))

    def compile(self) -> None:
        super().compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
