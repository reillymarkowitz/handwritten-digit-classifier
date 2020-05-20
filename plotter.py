import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot(accuracy, validation_accuracy):
        plt.plot(accuracy)
        plt.plot(validation_accuracy)
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()
