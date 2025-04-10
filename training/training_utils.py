import random
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch_geometric.data import Batch
import pickle
import re
from torchrl.collectors import MultiSyncDataCollector
import json
import os
import numpy as np

from env_creation.utils import compute_metric_stats

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



def evaluate_actor(test_env_func, actor, hp, writer, epoch = 0, main_tag = "eval", pickle_batch_data = False, batch_dir = None):
    # Create test collector
    test_collector = MultiSyncDataCollector(
        create_env_fn=test_env_func,
        policy=actor,
        frames_per_batch=hp["EVALS_PER_ENV"] * len(test_env_func),
    )

    print("Beginning Evaluation")
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = test_collector.next()
        if isinstance(td["next", "data"], list):
            flattened = [item for sublist in td["next", "data"] for item in sublist]
        else:
            flattened = [td["next", "data"]]
        new_batch = Batch.from_data_list(flattened)
        for context_id in new_batch["context_id"].unique():
            id_batch = Batch.from_data_list(new_batch[new_batch["context_id"] == context_id])
            log_info = {
                f"{main_tag}/{context_id}/Mean Backlog": id_batch["backlog"].mean(),
                f"{main_tag}/{context_id}/Mean Reward Context {context_id}": id_batch["reward"].mean(),
                # f"{main_tag}/{context_id}/BP Normalized Backlog Context {context_id}": id_batch["backlog_bp_norm"].mean(),
                # f"{main_tag}/{context_id}/SP Normalized Backlog Context {context_id}": id_batch["backlog_sp_norm"].mean(),
            }
            # from id_batch, get all keys with "norm" in them
            for key in id_batch.keys():
                if "norm" in key:
                    log_info[f"{main_tag}/{context_id}/{key}"] = id_batch[key].mean()
            for key, value in log_info.items():
                print(f"{key}: {value:.2f}")
                writer.add_scalar(key, value, epoch)
    if pickle_batch_data:
        if batch_dir is None:
            batch_dir = os.path.join(writer.log_dir, "eval_batches")
        pickle.dump(new_batch, file = open(os.path.join(batch_dir, f"eval_batch_{epoch}.pkl"), "wb"))
    test_collector.shutdown()

def extract_integer_from_string(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None




def create_training_and_test_contexts(context_set_dir,
                                      run_dir,
                                      training_context_indices = None,
                                      num_training_contexts = None,
                                      ):
    """
    From a context folder, create training and test contexts
    Returns two lists of context sets that can be used for training and testing


    :param context_set_dir: str, path to context set folder
    :param run_dir: str, path to run directory where the training and test contexts will be saved
    :param training_contexts: list[int], training contexts
    :param num_training_contexts: int, number of training contexts
    :param num_test_contexts: int, number of test contexts
    :return: list, training_contexts, test_contexts
    """

    if training_context_indices is None and num_training_contexts is None:
        raise ValueError("Either training_contexts or num_training_contexts must be provided")

    if training_context_indices is not None and num_training_contexts is not None:
        raise ValueError("Only one of training_contexts or num_training_contexts must be provided")

    if training_context_indices is not None:
        num_training_contexts = len(training_context_indices)

    context_set = os.listdir(context_set_dir)
    # filter out all non-env files
    context_set = [context for context in context_set if "Env" in context]

    if training_context_indices is None:
        training_contexts = np.random.choice(context_set, num_training_contexts, replace=False)
    else: # set training contexts based on input training contexts
        training_contexts = []
        for context in context_set:
            context_id = int(context.removesuffix(".json").split("_")[-1].removeprefix("Env"))
            if context_id in training_context_indices:
                training_contexts.append(context)


    # remove training contexts from context set to create test contexts
    test_contexts = [context for context in context_set if context not in training_contexts]

    # save training contexts to run  and compute metric stats
    training_metrics = []
    training_contexts_dir = os.path.join(run_dir, "training_contexts")
    os.makedirs(training_contexts_dir, exist_ok=True)
    for context in training_contexts:
        with open(os.path.join(context_set_dir, context), 'r') as f:
            context_config = json.load(f)
        training_metrics.append(context_config['metrics'])
        with open(os.path.join(training_contexts_dir, context), 'w') as f:
            json.dump(context_config, f)
    training_contexts_stats = compute_metric_stats(training_metrics)
    with open(os.path.join(training_contexts_dir, "context_set_stats.json"), 'w') as f:
        json.dump(training_contexts_stats, f)

    # save test contexts to run dir and compute/save metric stats
    test_metrics = []
    test_contexts_dir = os.path.join(run_dir, "test_contexts")
    os.makedirs(test_contexts_dir, exist_ok=True)
    for context in test_contexts:
        with open(os.path.join(context_set_dir, context), 'r') as f:
            context_config = json.load(f)
        test_metrics.append(context_config['metrics'])
        with open(os.path.join(test_contexts_dir, context), 'w') as f:
            json.dump(context_config, f)
    if len(test_metrics) > 0:
        test_contexts_stats = compute_metric_stats(test_metrics)
        with open(os.path.join(test_contexts_dir, "context_set_stats.json"), 'w') as f:
            json.dump(test_contexts_stats, f)

    # compute delta between all training context stats and all test context stats
        delta_stats = {}
        for key in training_contexts_stats:
            delta_stats[key] = {}
            for key2 in training_contexts_stats[key]:
                delta_stats[key][key2] = abs(training_contexts_stats[key][key2] - test_contexts_stats[key][key2])

        # print
        print("Created training and test contexts")
        print(f"Training context dir: {training_contexts_dir}")
        print(f"Test context dir: {test_contexts_dir}")
        # print the stats for training, test, and the difference in column format
        print("Metric | Stat | Training | Test | Delta:")
        for main_key, sub_dict in delta_stats.items():
            for sub_key, value in sub_dict.items():
                print(f"{main_key} | {sub_key} | {training_contexts_stats[main_key][sub_key]} | {test_contexts_stats[main_key][sub_key]} | {value}")


    # return folders for training and test contexts
    return training_contexts_dir, test_contexts_dir