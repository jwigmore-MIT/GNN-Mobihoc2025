import torch
import os
import warnings
import json
from env_sim.EnvSampler import make_env_mcmh_ts, make_env_mcmh_bp

import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial

now = datetime.now()
dt_string = now.strftime("%b%d_%H-%M")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import platform
opsys = platform.system()
if opsys == "Windows":
    PROJECT_DIR = "C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\"
else:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1])) # needed for RuntimeError: received 0 items of ancdata
    PROJECT_DIR = "/home/jwigmore/PycharmProjects/GDRL4Nets"


def evaluate_env(env_id, env_config, actor, hp, routing_paradigm, num_rollouts=3, max_steps=1000):
    """Function to evaluate a single environment"""
    make_env_func = make_env_mcmh_bp if routing_paradigm == "bp" else make_env_mcmh_ts
    env_results = []
    norm_env_results = []
    norm_factor = 1
    # get normalization factor
    try:
        if "baselines" in env_config:
            if hp["REWARD"] == "negative_backlog":
                if routing_paradigm == "bp":
                    norm_factor = - env_config["baselines"]["SPBP"]["backlog"]
                elif routing_paradigm == "randomized":
                    norm_factor = - env_config["baselines"]["SPTS"]["backlog"]
            if hp["REWARD"] == "negative_power_throughput":
                if routing_paradigm == "bp":
                    for key in env_config["baselines"].keys():
                        if "DPP" in key:
                            norm_factor = - env_config["baselines"][key]["power_throughput"]
    except:
        pass
    if hp["REWARD"] == "negative_power_throughput":
        env_config["action_func"] = "bpi"
    for n in range(num_rollouts):
        seed = int(env_config["seed"] + n)

        env = make_env_func(env_config, seed=seed)
        env_rep = env.get_rep(features=hp["FEATURES"])
        actor_output = actor(env_rep)

        if routing_paradigm == "randomized":
            env.W = actor_output["W"].squeeze()
        elif routing_paradigm == "bp":
            env.set_bias(actor_output["W"].squeeze())


        rollout = env.rollout(max_steps=max_steps)

        if hp["REWARD"] == "negative_backlog":
            score = - rollout["next", "backlog"].mean().item()
        elif hp["REWARD"] == "negative_power_throughput":
            power_mean = rollout["next", "power"].mean().item()
            throughput = rollout["next", "departures"].sum().item() / rollout["arrivals"].sum().item()
            score = - (power_mean + hp["REWARD_ARGS"]["throughput_penalty"] * (throughput < hp["REWARD_ARGS"]["min_throughput"]))
        else:
            raise ValueError("Reward function not recognized")
        env_results.append(score)
        norm_env_results.append(score / norm_factor)



    return env_id, {"mean_score": float(np.mean(env_results)), "std_score": float(np.std(env_results)),
                    "mean_norm_score": float(np.mean(norm_env_results)), "std_norm_score": float(np.std(norm_env_results))}


def evaluate_model(run_path, model_name = None):
    ### Recreate the __main__ block, but with the run_path as an argument
    # Rollouts per environment
    hp = json.load(open(os.path.join(run_path, "hyperparameters.json"), 'r'))
    num_rollouts = hp.get("EVALS_PER_ENV", 3)
    max_steps = hp.get("MAX_STEPS", 1000)

    if model_name is None:
        model_num = hp.get("RL_EPISODES", 500) + 1 # should be the final model saved
        model_name = f"model_{model_num}.pt"

    routing_paradigm = hp["ROUTING_PARADIGM"]  # "BP"

    # Load the model(s)
    actor = torch.load(os.path.join(run_path, model_name))

    # Training Environments
    train_dir = "training_contexts"
    training_envs_files = os.listdir(os.path.join(run_path, train_dir))
    training_envs_configs = {}
    for file in training_envs_files:
        if "Env" not in file:
            continue
        with open(os.path.join(run_path, train_dir, file), 'r') as f:
            env_id = int(file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
            training_envs_configs[env_id] = json.load(f)

    # Test Environments
    test_dir = "test_contexts"
    test_envs_files = os.listdir(os.path.join(run_path, test_dir))
    if len(test_envs_files) > 0:
        test_envs_configs = {}
        for file in test_envs_files:
            if "Env" not in file:
                continue
            with open(os.path.join(run_path, test_dir, file), 'r') as f:
                env_id = int(file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
                test_envs_configs[env_id] = json.load(f)

        all_env_configs = {**training_envs_configs, **test_envs_configs}
    else:
        all_env_configs = training_envs_configs
    if "EVAL_ENVS" in  hp:
        all_env_configs = {env_id: all_env_configs[env_id] for env_id in hp["EVAL_ENVS"]}

    # Create a multiprocessing pool
    # Use number of CPUs available minus 2 // 2 to split cpus with another parellel process and to leave two core free for system tasks
    num_processes = max(1, (mp.cpu_count() - 2)//2)
    print(f"Using {num_processes} processes for parallel evaluation")

    # Create a partial function with fixed arguments
    evaluate_env_partial = partial(
        evaluate_env,
        actor=actor,
        hp=hp,
        routing_paradigm=routing_paradigm,
        num_rollouts=num_rollouts,
        max_steps=max_steps
    )

    # Create arguments for each environment
    env_args = [(env_id, env_config) for env_id, env_config in all_env_configs.items()]

    # Run evaluations in parallel
    with mp.Pool(processes=num_processes) as pool:
        # map_async preserves the order of results
        results = pool.starmap(evaluate_env_partial, env_args)

    # Collect results
    all_results = dict(results)



    # Separate training and test results
    training_results = {env_id: all_results[env_id] for env_id in training_envs_configs.keys() if env_id in all_results}
    if len(test_envs_files) > 0:
        test_results = {env_id: all_results[env_id] for env_id in test_envs_configs.keys() if env_id in all_results}
    else:
        test_results = {}

    # compute the mean of each metric for each result
    for results in [training_results, test_results, all_results]:
        if len(results) == 0:
            continue
        agg = {"mean_score": None, "std_score": None, "mean_norm_score": None, "std_norm_score": None}
        for key in agg.keys():
            agg[key] = np.mean([result[key] for result in results.values()])
        results["Set_Average"] = agg




    # Save the results in the run directory, in a folder called "evaluation"
    eval_dir = os.path.join(run_path, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"training_eval_{dt_string}.json"), 'w') as f:
        json.dump(training_results, f)
    if len(test_results) > 0:
        with open(os.path.join(eval_dir, f"test_eval_{dt_string}.json"), 'w') as f:
            json.dump(test_results, f)

    return all_results




if __name__ == "__main__":
    # Rollouts per environment
    num_rollouts = 3
    max_steps = 1000

    # run_path = "experiments/GNNBiasedBackpressureDevelopment/saved_runs/Mar21_16-35-23_cnrg-lab-4gpu.mit.edu_randomized_routing_config6_meanIaAgg"
    run_path = "experiments/GNNBiasedBackpressureDevelopment/saved_runs/Mar23_15-09-01_cnrg-lab-4gpu.mit.edu_randomized_routing_config8_ActuallynoIA"
    model_name = "model_501.pt"
    hp = json.load(open(os.path.join(PROJECT_DIR, run_path, "hyperparameters.json"), 'r'))

    routing_paradigm = hp["ROUTING_PARADIGM"]  # "BP"

    # Load the model(s)
    actor = torch.load(os.path.join(PROJECT_DIR, run_path, model_name))

    # Training Environments
    train_dir = "training_contexts"
    training_envs_files = os.listdir(os.path.join(PROJECT_DIR, run_path, train_dir))
    training_envs_configs = {}
    for file in training_envs_files:
        if "Env" not in file:
            continue
        with open(os.path.join(PROJECT_DIR, run_path, train_dir, file), 'r') as f:
            env_id = int(file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
            training_envs_configs[env_id] = json.load(f)

    # Test Environments
    test_dir = "test_contexts"
    test_envs_files = os.listdir(os.path.join(PROJECT_DIR, run_path, test_dir))
    test_envs_configs = {}
    for file in test_envs_files:
        if "Env" not in file:
            continue
        with open(os.path.join(PROJECT_DIR, run_path, test_dir, file), 'r') as f:
            env_id = int(file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
            test_envs_configs[env_id] = json.load(f)

    all_env_configs = {**training_envs_configs, **test_envs_configs}

    # Create a multiprocessing pool
    # Use number of CPUs available minus 1 to leave one core free for system tasks
    num_processes = max(1, mp.cpu_count() - 2)
    print(f"Using {num_processes} processes for parallel evaluation")

    # Create a partial function with fixed arguments
    evaluate_env_partial = partial(
        evaluate_env,
        actor=actor,
        hp=hp,
        routing_paradigm=routing_paradigm,
        num_rollouts=num_rollouts,
        max_steps=max_steps
    )

    # Create arguments for each environment
    env_args = [(env_id, env_config) for env_id, env_config in all_env_configs.items()]

    # Run evaluations in parallel
    with mp.Pool(processes=num_processes) as pool:
        # map_async preserves the order of results
        results = pool.starmap(evaluate_env_partial, env_args)

    # Collect results
    all_results = dict(results)

    # Separate training and test results
    training_results = {env_id: all_results[env_id] for env_id in training_envs_configs.keys()}
    test_results = {env_id: all_results[env_id] for env_id in test_envs_configs.keys()}

    # Save the results in the run directory, in a folder called "evaluation"
    eval_dir = os.path.join(PROJECT_DIR, run_path, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"training_eval_{dt_string}.json"), 'w') as f:
        json.dump(training_results, f)
    with open(os.path.join(eval_dir, f"test_eval_{dt_string}.json"), 'w') as f:
        json.dump(test_results, f)