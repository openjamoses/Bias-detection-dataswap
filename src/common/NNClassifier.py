from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow._api.v2.compat.v1 as tf

class NNClassifier:
    def __init__(self, input_shape=7, output_shape=2):
        #disable_eager_execution()
        #tf.disable_v2_behavior()
        self.model = Sequential()
        self.model.add(Dense(500, activation='relu', input_dim=input_shape))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)
    def fit(self, x_train, y_train, epoch=2):
        # build the model
        self.model.fit(x_train, y_train, epochs=epoch)
        return self.model
    def predict(self, x_test):
        #pred = self.model.predict(x_test)
        #y_classes = pred.argmax(axis=-1)
        #print("Class: ", y_classes)
        return self.model.predict(x_test)
    def evaluate(self, x_test, y_test, verbose=0):
        return self.model.evaluate(x_test, y_test, verbose=verbose)
