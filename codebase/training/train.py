import sys
import os

# Add the PROJECT directory to the Python path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message="co_lnotab is deprecated, use co_lines instead")
from collections import defaultdict

from env_sim.EnvSampler import EnvSampler, make_env_mcmh_ts, make_env_mcmh_bp
from env_sim.NetworkRunner import create_network_runner
from evaluation.eval_training_model_mp import evaluate_model

import json
from models.DeeperIntranodeAggGNN import DeeperIntranodeAggGNN

from models.wrappers import NormalWrapper
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal, TruncatedNormal
import torch.optim as optim
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.collectors import MultiSyncDataCollector
from torch_geometric.data import Batch
from .training_utils import seed_all, create_training_and_test_contexts
from copy import deepcopy
from torchrl.data import ReplayBuffer, ListStorage
import argparse
import pickle
# Get the operating system
import platform
opsys = platform.system()
# get the number of cpus
if opsys == "Windows":
    pass
else:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1])) # needed for RuntimeError: received 0 items of ancdata

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Set printing options for torch
torch.set_printoptions(sci_mode=False, precision=3)


def log_global_hparams(writer):
    hparams = {var_name: var_value for var_name, var_value in globals().items() if isinstance(var_value, (int, float, str))}
    writer.add_hparams(hparams, {})

def load_hyperparameters(args):
    """
    Cases:
        1. args.config is None - then either choose windows or linux config
        2. args.config is string - then load that config

    for both cases, set hp["config"] to the config file name


    :param args:
    :return:
    """
    if args.config is None:
        # Default Windows
        if opsys == "Windows":
            config_path = "Exp1_minQ_spdaIA_st_TwoPathEnv1.json"
        else:
            config_path = "linux_ts_config.json"
    else:
        config_path = args.config

    rel_config_path = "configs/" + config_path

    with open(rel_config_path, "r") as f:
        hp = json.load(f)

    hp["config"] = config_path

    for key, value in vars(args).items():
        if value is not None:
            hp[key] = value

    hp["TOTAL_FRAMES"] = hp["ROLLOUTS_PER_ENV"] * hp["RL_EPISODES"]

    if opsys == "Windows":
        hp["RL_EPISODES"] = 10

    return hp

def parse_input():
    parser = argparse.ArgumentParser(description='Run title input')
    parser.add_argument('--run_title', type=str, required=False, help='Title for the run', default="")
    parser.add_argument('--config', type=str, required=False, help='Path to the configuration file', default=None)
    # Add optional arguments for each hyperparameter
    parser.add_argument('--ROLLOUTS_PER_ENV', type=int, help='Number of rollouts per environment')
    parser.add_argument('--BUFFER_HISTORY', type=int, help='Buffer history size')
    parser.add_argument('--RL_EPOCHS', type=int, help='Number of RL epochs')
    parser.add_argument('--NEW_BATCH_SAVE_FREQ', type=int, help='Frequency of saving new batches')
    parser.add_argument('--TOTAL_FRAMES', type=int, help='Total number of frames')
    parser.add_argument('--MAX_STEPS', type=int, help='Maximum number of steps')
    parser.add_argument('--EVALS_PER_ENV', type=int, help='Number of evaluations per environment')
    parser.add_argument('--SUP_EPOCHS', type=int, help='Number of supervised epochs')
    parser.add_argument('--TRAINING_CONTEXTS', type=int, nargs='+', help='List of training contexts')
    parser.add_argument('--SAVE_IMAGES', type=bool, help='Whether to save images')
    parser.add_argument('--SP_ALPHA', type=int, help='SP alpha value')
    parser.add_argument('--GNN_LAYERS', type=int, help='Number of GNN layers')
    parser.add_argument('--GNN_HIDDEN', type=int, help='Number of GNN hidden units')
    parser.add_argument('--GNN_AGGR', type=str, help='GNN aggregation method')
    parser.add_argument('--INTRANODE_ATTN', type=bool, help='Whether to use intranode attention')
    parser.add_argument('--SEED', type=int, help='Random seed')
    parser.add_argument('--SCALE_BIAS', type=int, help='Scale bias value')
    parser.add_argument('--PPO_EPSILON', type=float, help='PPO epsilon value')
    parser.add_argument('--RL_LR', type=float, help='RL learning rate')
    parser.add_argument('--LINK_REWARDS', type=str, nargs='+', help='List of link rewards')
    parser.add_argument('--SUPERVISED_PRETRAINING', type=bool, help='Whether to use supervised pretraining')
    parser.add_argument('--PRE_TRAINING_PERFORMANCE', type=bool, help='Whether to evaluate pre-training performance')
    parser.add_argument('--SUP_LR', type=float, help='Supervised learning rate')
    args = parser.parse_args()
    return args

def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm



def train():

    ## Collect user arguments from command line
    args = parse_input()

    hp = load_hyperparameters(args)

    run_title = hp["config"].split(".")[0]
    if hp.get("RUN_TITLE", None) is not None:
        run_title = run_title + "_" + hp["RUN_TITLE"].replace(" ", "_")
    if args.run_title is not None:
        run_title = run_title + "_" + args.run_title.replace(" ", "_")

    #print working directory
    print(f"Working Directory: {os.getcwd()}")

    writer = SummaryWriter(comment=run_title)
    # log_global_hparams(writer)

    # get run directory
    run_dir = writer.log_dir
    print(f"Run Directory: {run_dir}")
    training_batch_dir = os.path.join(run_dir, "training_batches")
    os.makedirs(training_batch_dir, exist_ok=True)
    eval_batch_dir = os.path.join(run_dir, "eval_batches")
    os.makedirs(eval_batch_dir, exist_ok=True)

    # save hp to run_dir
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hp, f)


    # set all random seeds
    seed_all(hp["SEED"])

    # get clip bounds
    ppo_eps = torch.tensor([hp["PPO_EPSILON"]])
    clip_bounds = ((-ppo_eps).log1p(),
            ppo_eps.log1p())


    context_set_path = os.path.join(PROJECT_DIR, hp["CONTEXT_SET_PATH"])
    # check if context_set_path leads to a directory or a file
    if os.path.isdir(context_set_path): # if dir, create a context set dict from all contexts in the directory
        # Create training and context sets
        training_context_dir, test_context_dir = create_training_and_test_contexts(context_set_path,
                                                                                   run_dir,
                                                                                   training_context_indices=hp[
                                                                                       "TRAINING_CONTEXTS"])

        # get all files in the directory
        context_set_dict = {}
        for file in os.listdir(training_context_dir):
            if "Env" in file:
                context_id = int(file.removesuffix(".json").split("_")[-1].removeprefix("Env"))
                context_set_dict[context_id] = json.load(open(os.path.join(training_context_dir, file), 'r'))
    else:
        context_set_dict = json.load(open(context_set_path, 'r'))

    make_env_funcs = {"randomized": make_env_mcmh_ts,
                      "bp": make_env_mcmh_bp}

    make_env_keywords = {"max_backlog": hp.get("MAX_BACKLOG", None),
                         "action_func": hp.get("ACTION_FUNC", None),}

    env_sampler = EnvSampler(context_set_dict,
                             make_env_keywords=make_env_keywords,
                             env_generator_seed=0,
                             make_env_fn=make_env_funcs[hp["ROUTING_PARADIGM"]])

    # modify training_envs if necessary
    training_contexts = []
    for context_id in hp["TRAINING_CONTEXTS"]:
        if context_id not in env_sampler.context_dicts.keys():
            print(f"Training Context {context_id} not in env generator. Removing it from training")
        else:
            training_contexts.append(context_id)


    base_env = env_sampler.sample()
    network_specs = base_env.get_rep(hp["FEATURES"])


    model = DeeperIntranodeAggGNN(
        node_channels = network_specs["X"].shape[-1],
        edge_channels = network_specs["edge_attr"].shape[-1],
        hidden_channels =hp["GNN_HIDDEN"],
        edge_message=True,
        num_layers = hp["GNN_LAYERS"],
        output_channels=2,
        output_activation=None,
        edge_decoder=True,
        aggr = hp["GNN_AGGR"],
        intranode_aggregation = hp.get("INTRANODE_AGG", "spda"),
        intranode_aggregation_kwargs = hp.get("INTRANODE_AGG_KWARGS", {}),
        conv_output_func = hp.get("GNN_CONV_OUTPUT_FUNC", "relu"),
        internal_weights = hp.get("INTERNAL_WEIGHTS", True),
    )


    norm_module = NormalWrapper(model, scale_bias=hp["SCALE_BIAS"])

    prob_out_keys = ["W"]  #if hp["ROUTING_PARADIGM"] == "bp" else ["Wu"]
    prob_actor = ProbabilisticActor(norm_module,
                               in_keys=["loc", "scale"],
                               out_keys=prob_out_keys,
                               distribution_class=IndependentNormal if hp.get("ROUTING_PARADIGM", "bp") == "bp" else TruncatedNormal,
                               distribution_kwargs={} if hp.get("ROUTING_PARADIGM", "bp") == "bp" else {"low": 0, "high": 100, "tanh_loc": False},
                               return_log_prob=True,
                               default_interaction_type=ExplorationType.DETERMINISTIC
                               )

    actor = prob_actor


    # test a single forward pass
    # output = actor(TensorDict({"X": network_specs["X"], "edge_index": network_specs["edge_index"], "edge_attr": network_specs["edge_attr"]}))



    static_kwargs = {"features": hp["FEATURES"],"max_steps": hp["MAX_STEPS"], "type": hp["ROUTING_PARADIGM"], "reward": hp["REWARD"], "reward_args": hp.get("REWARD_ARGS", {})}

    # Training Environment Initialization
    training_env_fn = [
        lambda context_id=context_id: create_network_runner(env=env_sampler.sample(context_id), **static_kwargs)
        for context_id in training_contexts]

    # per context_id replay buffers
    if hp.get("BUFFER_HISTORY", 0) > 0:
        replay_buffers = {
            context_id: ReplayBuffer(
                storage=ListStorage(max_size=hp["BUFFER_HISTORY"] * hp["ROLLOUTS_PER_ENV"]),
                collate_fn=lambda x: x,
                generator=torch.Generator().manual_seed(hp["SEED"])
            )
            for context_id in training_contexts
        }


    training_collector = MultiSyncDataCollector(
        create_env_fn=training_env_fn,
        policy=actor,
        frames_per_batch=hp["ROLLOUTS_PER_ENV"]*len(training_env_fn),
    )


    # Create parameter groups
    param_groups = [
        {"params": [param for name, param in model.named_parameters() if name != "scale_param"], "lr": hp.get("PPO_LR", 0.0005)},
        # {"params": [model.scale_param], "lr": scale_param_lr}
    ]

    if hasattr(model, "scale_param") and model.scale_param is not None:
        param_groups.append({"params": [model.scale_param], "lr": hp.get("SCALE_PARAM_LR", 0.0005)})

    # Create the ppo_optimizer with parameter groups
    ppo_optimizer = optim.Adam(param_groups)

    # For saving best models
    best_model_name = None
    best_score = -1e8

    # training loop
    for eps, all_new_samples in enumerate(training_collector):
        print(f"------Episode {eps} -----")
        all_log_info = {}
        n_envs = all_new_samples.shape[0]

        # Create batches and update replay buffers for each context_id
        batches = {}
        for n in range(n_envs):  # process each context_id separately
            new_samples = all_new_samples[n]
            context_id = new_samples["next", "context_id"][0].item()
            # print(f"{new_samples["loc"][0,-3:]}  \n {new_samples["scale"][0,-3:]}")
            # Deal with nested lists
            if isinstance(new_samples["next", "data"], list): #
                if all(isinstance(i, list) for i in new_samples["next", "data"]):
                    flattened = [item for sublist in new_samples["next", "data"] for item in sublist]
                else:
                    flattened = new_samples["next", "data"]
            else:
                flattened = [new_samples["next", "data"]]

            # Save batch periodically
            new_batch = Batch.from_data_list(flattened)
            if eps % hp["NEW_BATCH_SAVE_FREQ"] == 0:
                pickle.dump(new_batch,
                            file=open(os.path.join(training_batch_dir, f"new_batch_{eps}_{context_id}.pkl"), "wb"))

            # Update replay buffer and prepare batch for training
            if hp.get("BUFFER_HISTORY", 0) > 0 and len(replay_buffers[context_id]) > 0:
                update_samples = deepcopy(flattened)
                update_samples.extend(replay_buffers[context_id][:])
            else:
                update_samples = flattened

            if hp.get("BUFFER_HISTORY", 0) > 0:
                replay_buffers[context_id].extend(flattened)

            context_batch = Batch.from_data_list(update_samples)



            # Store for later use in mini-epochs
            batches[context_id] = {
                "batch": context_batch,
                "new_batch": new_batch
            }



        # training metrics
        log_probs = defaultdict(list)
        clip_fractions = defaultdict(list)
        ratios = defaultdict(list)
        gains = defaultdict(list)
        ppo_losses = defaultdict(list)
        entropy_losses = defaultdict(list)
        # Training mini-epochs
        for mini_epoch in range(hp["PPO_EPOCHS"]):

            total_ppo_loss = torch.tensor(0.0)


            # Process each context_id within this mini-eps
            for context_id, batch_data in batches.items():
                batch = batch_data["batch"]

                # Input to the actor's distribution
                dist_td = TensorDict({"X": batch.X, "edge_index": batch.edge_index, "edge_attr": batch.edge_attr})

                # Get distribution and compute losses
                dist = prob_actor.get_dist(dist_td)

                # Routing coefficients from sampled data (B,M,K) or (M,K) if B=1
                W = batch[prob_out_keys[0]]
                if W.shape != dist.base_dist.loc.shape:
                    W.unsqueeze_(-1)
                # Log probability of each routing coefficient (B,M,K)
                log_prob = dist.log_prob(W)

                # Entropy of each component of distribution (B,M, K)
                entropy = dist.entropy()

                # Log ratio of the probability of the sampled data to the probability of the batch data
                log_weight = (log_prob - batch.sample_log_prob).squeeze()

                # Gain from the link rewards
                gain1 = log_weight.exp() * batch["link_rewards"]

                # Clipped Gain Computation
                log_weight_clip = log_weight.clamp(*clip_bounds)
                clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
                ratio = log_weight_clip.exp()
                gain2 = ratio * batch["link_rewards"]

                # Taking the minimum of the two gains for trust-region optimization approximation (PPO Clip loss)
                gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]

                # Add the loss to
                ppo_loss = -gain.mean()*hp.get("PPO_LOSS_FACTOR", 1.0)

                # If ppo_loss is extremely large, or NaN, skip this mini-epoch
                if torch.isnan(ppo_loss) or ppo_loss.abs() > 1e3:
                    print(f"Skipping mini-epoch {mini_epoch} for env id {context_id} due to large loss = {ppo_loss.item()}")
                    ppo_loss = torch.tensor(0.0)

                # Compute entropy loss
                entropy_loss = - hp.get("ENTROPY_BONUS", 0) * entropy.mean()

                total_ppo_loss += ppo_loss + entropy_loss

                # Collect metrics for this context_id in this mini-eps
                log_probs[context_id].append(log_prob.mean())
                clip_fractions[context_id].append(clip_fraction)
                ratios[context_id].append(ratio.mean())
                gains[context_id].append(gain.mean())
                ppo_losses[context_id].append(ppo_loss)
                entropy_losses[context_id].append(entropy_loss)


            # Backpropagation
            ppo_optimizer.zero_grad()
            total_ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ppo_optimizer.step()


        # Log metrics for each context_id
        for context_id, batch_data in batches.items():
            new_batch = batch_data["new_batch"]

            log_info = {
                # "Loss": torch.stack(total_losses_all).mean().item(),
                "PPO Loss": torch.stack(ppo_losses[context_id]).sum().item() if len(ppo_losses[context_id]) > 0 else 0,
                "Entropy Loss": torch.stack(entropy_losses[context_id]).sum().item() if len(entropy_losses[context_id]) > 0 else 0,
                "Average Link Loss": torch.stack(gains[context_id]).mean().item() if len(gains[context_id]) > 0 else 0,
                "Mean Backlog": new_batch["backlog"].mean(),
                "Mean Reward": new_batch["reward"].mean(),
                "Mean Power": new_batch["power"].mean(),
                "Mean Reward Baseline Ratio": new_batch["reward_baseline_ratio"].mean() if "reward_baseline_ratio" in new_batch.keys() else 0,
                "Max Reward": new_batch["reward"].max(),
                "Min Reward": new_batch["reward"].min(),
                "Std Reward": new_batch["reward"].std(),
                "Sample Log Prob": new_batch["sample_log_prob"].mean(),
                "Batch Log Prob": torch.stack(log_probs[context_id]).mean().item() if len(log_probs[context_id]) > 0 else 0,
                "Clip Fraction": torch.stack(clip_fractions[context_id]).mean().item() if len(clip_fractions[context_id]) > 0 else 0,
                "Ratio Mean": torch.stack(ratios[context_id]).mean().item() if len(ratios[context_id]) > 0 else 0,
                "Exploration_Scale": new_batch["scale"].mean().item(),
            }

            # Write to tensorboard and print key metrics
            for key, value in log_info.items():
                context_key = f"{key}/{context_id}"
                writer.add_scalar(context_key, value, eps)
                if key in ["Mean Backlog", "Mean Reward", "Loss"]:
                    print(f"{context_key}: {value:.2f}")

            all_log_info[context_id] = log_info

        # Log average metrics across all context_ids
        for key in all_log_info[training_contexts[0]].keys():
            writer.add_scalar(key, sum([all_log_info[context_id][key] for context_id in training_contexts]) / len(
                training_contexts), eps)

        # Save model periodically
        if (eps + 1) % hp.get("MODEL_SAVE_FREQ", 100) == 0:
            # model name should be model_{mean_training_reward}_{eps}.pt
            mean_training_reward = int((sum([all_log_info[context_id]['Mean Reward'] for context_id in training_contexts]) / len(training_contexts)).item())
            model_name = f"model_{mean_training_reward}_{eps + 1}.pt"
            model_path = os.path.join(run_dir, model_name)
            torch.save(actor, model_path)

            if mean_training_reward > best_score:
                best_score = mean_training_reward
                best_model_name = model_name
                print(f"New Best Model Saved: {model_path}")

        if eps >= hp["RL_EPISODES"]:
            break

    training_collector.shutdown()

    print("Finished Training")
    # Save the model
    model_name = f"model_{eps + 1}.pt"
    if best_model_name is None:
        best_model_name = model_name
    model_path = os.path.join(run_dir, model_name)
    torch.save(actor, model_path)
    print(f"Model saved to {model_path}")

    # Do Model evaluation

    # get the absolute path to the run_dir
    abs_run_path = os.path.abspath(run_dir)
    eval_results = evaluate_model(abs_run_path, model_name = best_model_name)

    #TODO Log eval_results
    for env_id, results in eval_results.items(): #env_id, {"mean_score":..., "std_score": ..., "mean_norm_score": ..., "std_norm_score": ...}),
        for key, value in results.items():
            writer.add_scalar(f'eval/{env_id}/{key}', value, eps+1)
    writer.close()

    return eval_results["Set_Average"]["mean_norm_score"]

if __name__ == "__main__":
    result = train()





