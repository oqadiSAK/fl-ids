import os
import tensorflow as tf
import flwr as fl
import utils.dataloader as dataloader
from model import Model

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = dataloader.get_data()
        self.model = Model(self.X_train.shape[1:])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, _):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.Y_train, epochs=1, batch_size=64)
        return self.model.get_weights(), len(self.X_train), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, _):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
