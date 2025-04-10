from utils import *
import numpy as np
import random
from env_sim.MultiClassMultihopTS import MultiClassMultiHopTS, create_link_weights
from env_sim.MultiClassMultihopBP import MultiClassMultiHopBP, create_sp_bias
import inspect
from create_random_networks import get_metrics

# set print options for torch
torch.set_printoptions(precision=2, sci_mode=False)





def get_network_baselines(network_config,
                          max_steps = 1000,
                          num_evals = 3,
                          policies = ["SPTS", "SPBP"],):
    # test the network eval_num times
    results = {}
    method_dict = {"CWTS": "capacity_weighted_shortest_path",
                   "SPTS": "shortest_path",
                   "SPBP2": "capacity_weighted_shortest_path",
                   "ICWTS": "inverse_capacity_weighted_shortest_path",}

    base_env = MultiClassMultiHopTS(**network_config)
    G = base_env.graphx
    metrics = get_metrics(network_config, G)
    network_config["metrics"] = metrics
    for poli in policies:
        results[poli] = []
        for n in range(num_evals):
            network_config["seed"] = n + 1
            if poli in ["BP", "SPBP", "SPBP2"]:
                network_config["action_func"] = None
                env = MultiClassMultiHopBP(**network_config)
                if poli == "SPBP":
                    bias, _ = create_sp_bias(env, weighted=True)
                    env.set_bias(bias)
                elif poli == "SPBP2":
                    weights = create_link_weights(env, method=method_dict["SPBP2"])
                    env.set_bias(weights)
            else:
                env = MultiClassMultiHopTS(**network_config)
                weights = create_link_weights(env, method=method_dict[poli])
                env.W = weights
            td = env.rollout(max_steps)

            if env.__str__() == "MultiClassMultiHopBP()":
                max_class_backlog = td["Q"].float().mean(dim=0).sum(dim=0).max(dim=0)[0]
            elif env.__str__() == "MultiClassMultiHopTS()":
                max_class_backlog = td["L_class_count"].float().mean(dim=0).sum(dim=0).max(dim=0)[0]
            else:
                max_class_backlog = torch.zeros(1)

            results[poli].append(TensorDict({
                "mean_backlog": td["backlog"].mean(),
                "departures": td["next", "departures"].sum(),
                "arrivals": td["arrivals"].sum(),
                "throughput": td["next", "departures"].sum() / td["arrivals"].sum(),
                "max_class_backlog": max_class_backlog,
            }))


    avg_results = {}
    std_results = {}
    for poli in results:
        avg_results[poli] = TensorDict.stack(results[poli]).mean()
        std_results[poli] = TensorDict.stack(results[poli]).std()

    network_config["baselines"] = {}
    network_config["baselines"]["max_steps"] = max_steps
    if metrics:
        # print all metrics for the network
        for key, value in network_config["metrics"].items():
            print(f"{key}: {value}")
    print("Throughput and Backlog for each policy")
    for poli in policies:
        network_config["baselines"][poli] = {
            "backlog": round(avg_results[poli]["mean_backlog"].item(), 2),
            "throughput": round(avg_results[poli]["throughput"].item(), 3),
            "backlog_std": round(std_results[poli]["mean_backlog"].item(), 2),
            "max_class_backlog": round(avg_results[poli]["max_class_backlog"].item(), 2),
        }
        network_config["metrics"][f"{poli}_backlog"] = round(avg_results[poli]["mean_backlog"].item(), 2)
        network_config["metrics"][f"{poli}_throughput"] = round(avg_results[poli]["throughput"].item(), 3)
        network_config["metrics"][f"{poli}_max_class_backlog"] = round(avg_results[poli]["max_class_backlog"].item(), 2)
        # print the throughput and backlog
        print(f"{poli} : Throughput = {round(avg_results[poli]['throughput'].item(), 2)}"
              f"; Backlog = {round(avg_results[poli]['mean_backlog'].item(), 2)}"
              )

def get_environment_set_baselines_and_metric(env_set_path, max_steps = 1000, num_evals = 3, policies = ["SPTS", "SPBP"], env_ids = None):
    # load all files in directory
    env_set = os.listdir(env_set_path)
    env_set = [env for env in env_set if "Env" in env]

    # create new env set folder (remove the last /)
    new_env_set_path = env_set_path if env_set_path[-1] != "/" else env_set_path[:-1]
    new_env_set_path =  new_env_set_path + "_baselines/"
    os.makedirs(new_env_set_path, exist_ok = True)


    env_configs = {}
    for env_file in env_set:
        env_id = int(env_file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
        if env_ids is not None:
            if env_id not in env_ids:
                continue
        env_config = json.load(open(os.path.join(env_set_path, env_file), 'r'))
        print(f"Testing Environment {env_id}")
        get_network_baselines(env_config, max_steps = max_steps, num_evals = num_evals, policies = policies)
        # save the network config
        json.dump(env_config, open(os.path.join(new_env_set_path, env_file), 'w'))
        env_configs[env_id] = env_config
    env_set_stats = compute_metric_stats([network_config["metrics"] for network_config in env_configs.values()])

    with open(os.path.join(new_env_set_path,
                           f"env_set_metrics.json"), 'w') as f:
        json.dump(env_set_stats, f)
    # print context set statistics
    print("Context Set Statistics")
    for key, value in env_set_stats.items():
        print(f"{key}: {value}")
    return env_configs, env_set_stats


if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_set_path = os.path.join(PROJECT_DIR, 'context_sets/env_set_c')
    env_configs, env_set_stats = get_environment_set_baselines_and_metric(env_set_path, max_steps = 1000, num_evals = 3, policies = ["SPTS", "SPBP"])
