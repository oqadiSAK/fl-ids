import os
import flwr as fl
import utils.data_loader as data_loader
import utils.model_loader as model_loader

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = model_loader.get_model(self.X_train.shape[1:])

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
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_numpy_client(server_address=server_address, client=Client())