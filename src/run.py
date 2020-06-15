from classifier import DigitClassifier
from data_provider import DataProvider
from plotter import Plotter

classifier = DigitClassifier()
provider = DataProvider(classifier.image_size, classifier.num_digits)

x_train, y_train = provider.get_training_data()
training_result = classifier.train_model(x_train, y_train)

history = training_result.history
Plotter.plot(history['accuracy'], history['val_accuracy'])

x_test, y_test = provider.get_test_data()
classifier.evaluate_model(x_test, y_test)
