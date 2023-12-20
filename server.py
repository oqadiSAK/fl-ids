import flwr as fl
import os

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    return {k: v / total_examples for k, v in federated_metrics.items()}

def get_server_strategy():
    return fl.server.strategy.FedAvg(
            min_fit_clients=3,
            min_evaluate_clients = 3,
            min_available_clients=3,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=get_server_strategy(),
        config=fl.server.ServerConfig(num_rounds=3),
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")
