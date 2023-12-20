import flwr as fl
import client
import os
from server import weighted_average 

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_client():
    return client.Client()

if __name__ == "__main__":
    history = fl.simulation.start_simulation(
        client_fn=create_client,
        num_clients=2,
        strategy=fl.server.strategy.FedAvg(
            min_available_clients=2,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        ),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
