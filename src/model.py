from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History


class Model(Sequential):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.construct()
        self.compile()

    def construct(self) -> None:
        hidden_layers = [2500, 2000, 1500, 1000, 500]

        self.add(Dense(units=hidden_layers[0], activation='sigmoid', input_shape=(self.input_size,)))

        for layer in hidden_layers[1:]:
            self.add(Dense(units=layer, activation='sigmoid'))

        self.add(Dense(units=self.output_size, activation='softmax'))

    def compile(self) -> None:
        super().compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train) -> History:
        return self.fit(x_train, y_train, batch_size=128, epochs=200, validation_split=.1)
