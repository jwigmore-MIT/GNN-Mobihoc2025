from copy import deepcopy
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import networkx as nx

import numpy as np
import torch
from tensordict import TensorDict, merge_tensordicts
from .MultiClassMultihopBP import MultiClassMultiHopBP
import json
from torchrl.data import Composite, Bounded, Unbounded, Binary, NonTensorSpec, NonTensor
from tensordict import NonTensorData
import math

from torchrl.envs import (
    EnvBase,
)

import multiprocessing
from torch_geometric.data import Data

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROUTING_KEYS = ["data", "loc", "logits", "sample_log_prob", "scale", "Wu"]

def create_network_runner(env: EnvBase, features, max_steps: int = 1000, actor = None, type = "bp", reward = "negative_backlog", seed = 0, **kwargs):
    if type == "bp":
        return NetworkRunnerBP(env = env, features = features, max_steps = max_steps, actor = actor, reward = reward, seed = seed, **kwargs)
    elif type == "randomized":
        return NetworkRunnerRouting(env = env, features = features, max_steps = max_steps, actor = actor, seed = seed, reward = reward, **kwargs)

class NetworkRunnerBP(EnvBase): # Modeled after torhrl EnvBase
    def __init__(self, env: EnvBase, features: List[str],  max_steps: int = 1000, actor = None, reward = "negative_backlog", alpha = 0.1,
                 **kwargs):
        super().__init__()
        self.env = env
        self.features = features
        self.max_steps = max_steps
        self.actor = actor
        self.create_spec()

        # reward normalization parameters
        self.alpha = alpha

        # if negative_backlog, then self.reward = -backlog
        self.reward = reward
        self.reward_args = kwargs.get("reward_args",{})
        self.reward_normalizer = EWMARewardNormalizer(alpha=alpha)




    def create_spec(self):

        self.observation_spec = Composite({
            "data": NonTensor(shape = (1,), dtype = torch.float32),
            "Qavg": Unbounded(shape =-1),
        })
        self.action_spec = Composite({
            "bias": Unbounded(shape =-1)
        })


    def _reset(self, *args, **kwargs):
        return self.env.get_rep(self.features)

    def _set_seed(self, seed: Optional[int] = None):
        return

    def _step(self, td: TensorDict = None) -> TensorDict:
        return self.get_run(td)

    def get_run(self, tensordict = None):
        if tensordict is not None:
            if "data" in tensordict:
                Warning(" `data` already in td")
            if "W" in tensordict:
                self.env.set_bias(tensordict["W"])

        # initialize tensordict
        td = self.env.get_rep(self.features)

        if tensordict is not None and "W" in tensordict:
            self.env.set_bias(tensordict["W"])

        # collect rollout data
        rollout = self.env.rollout(max_steps = self.max_steps)

        # record metrics
        td["backlog"] = rollout["backlog"].mean()

        # get the routing coefficients
        td["W"] = self.env.bias.clone()

        td["power"] = rollout["next"].get("power", torch.tensor([0])).mean()

        td["throughput"] = rollout["next", "departures"].sum()/rollout["arrivals"].sum()

        # calculate reward
        # calculate reward
        reward_baseline_ratio = 1
        if self.reward == "negative_backlog":
            reward = -td["backlog"].mean()
            if "SPBP" in self.env.baselines.keys():
                reward_baseline_ratio = reward / self.env.baselines["SPBP"]["backlog"]
        elif self.reward == "negative_power_throughput":
            reward = -(td["power"] +
                       self.reward_args["throughput_penalty"]*
                       (td["throughput"] < self.reward_args["min_throughput"]))
            for key in self.env.baselines.keys():
                if "DPP" in key:
                    reward_baseline_ratio = reward / self.env.baselines[key]["power"]
        elif self.reward == "negative_max_class_backlog":
            reward = -rollout["Q"].float().mean(dim=0).sum(dim=0).max(dim=0)[0]
            if "SPTS" in self.env.baselines.keys():
                reward_baseline_ratio = reward / self.env.baselines["SPTS"]["backlog"]

        td["reward"] = reward
        td["reward_baseline_ratio"] = reward_baseline_ratio
        td["norm_reward"] = self.reward_normalizer.normalize(reward)
        td["link_rewards"] = td["norm_reward"] * torch.ones_like(td["W"])

        td["num_steps"] = len(rollout)

        # get data from tensordict
        if tensordict is not None:
            for key in tensordict.keys():
                if key in ROUTING_KEYS:
                    td[key] = tensordict[key]

        td["context_id"] = self.env.context_id

        # Create Data object
        graph = Data()
        for key in td.keys():
            if key == "data":
                pass
            else:
                try:
                    graph[key] = td[key].clone()
                except:
                    graph[key] = td[key]
        td["data"] = NonTensorData(graph)
        return td



class NetworkRunnerRouting(EnvBase): # Modeled after torhrl EnvBase
    def __init__(self, env: EnvBase,
                 features: List[str],
                 max_steps: int = 1000,
                 actor = None,
                 alpha = 0.1,
                 reward = "negative_backlog",
                 **kwargs):
        super().__init__()
        self.env = env
        self.features = features
        self.max_steps = max_steps
        self.actor = actor
        self.create_spec()

        # reward normalization parameters
        self.alpha = alpha

        # if negative_backlog, then self.reward = -backlog
        self.reward = reward

        self.reward_normalizer = EWMARewardNormalizer(alpha = alpha)



    def create_spec(self):
        self.observation_spec = Composite({
            "data": NonTensor(shape = (1,), dtype = torch.float32),
            "Qavg": Unbounded(shape =-1),
        })
        self.action_spec = Composite({
            "bias": Unbounded(shape =-1)
        })


    def _reset(self, *args, **kwargs):
        return self.env.get_rep(self.features)

    def _set_seed(self, seed: Optional[int] = None):
        return

    def _step(self, td: TensorDict = None) -> TensorDict:
        return self.get_run(td)

    def get_run(self, tensordict = None):
        # Check if "data" is already in td - means it contains a pyg Data object
        if tensordict is not None and "data" in tensordict:
            Warning(" `data` already in td")

        # initialize tensordict
        td = self.env.get_rep(self.features)

        # If there is a "bias" in td, set the bias to that value
        if tensordict is not None and "W" in tensordict:
            self.env.W = tensordict["W"].squeeze()


        # collect rollout data
        rollout = self.env.rollout(max_steps = self.max_steps)

        # record metrics
        td["backlog"] = rollout["backlog"].mean()

        td["power"] = rollout["next"].get("power", torch.tensor([0.0])).float().mean()

        td["throughput"] = rollout["next", "departures"].sum() / rollout["arrivals"].sum()

        # Get the utilize routing coefficients
        td["W"] = self.env.W.clone()

        # calculate reward
        reward_baseline_ratio = 1
        if self.reward == "negative_backlog":
            reward = -td["backlog"].mean()
            if "SPTS" in self.env.baselines.keys():
                reward_baseline_ratio = reward / self.env.baselines["SPTS"]["backlog"]
        elif self.reward == "negative_power_backlog":
            reward = -(td["power"] + td["backlog"]).mean()
        elif self.reward == "negative_max_class_backlog":
            reward = -rollout["L_class_count"].float().mean(dim=0).sum(dim=0).max(dim=0)[0]
            if "SPTS" in self.env.baselines.keys():
                reward_baseline_ratio = reward / self.env.baselines["SPTS"]["backlog"]


        td["reward"] = reward
        td["reward_baseline_ratio"] = reward_baseline_ratio
        td["norm_reward"] = self.reward_normalizer.normalize(reward)
        td["link_rewards"] = td["norm_reward"]*torch.ones_like(td["W"])

        td["Lavg"] = rollout["L_sizes"].float().mean(dim =0)
        td["L_class_count"] = rollout["L_class_count"].float().mean(dim =0)
        td["num_steps"] = len(rollout)

        # Get data from tensordict
        if tensordict is not None:
            for key in tensordict.keys():
                if key in ROUTING_KEYS:
                    td[key] = tensordict[key]

        # # Get baselines for normalization if available
        # for baseline in self.env.baselines.keys():
        #     if "std" in baseline or "steps" in baseline:
        #         continue
        #     td[f"{baseline}_norm"] = td["backlog"]/self.env.baselines[baseline]

        # Get performance of baseline algorithms
        td["sp_lta"] = self.env.baselines.get("sp_lta", 1)
        td["context_id"] = self.env.context_id

        # Create Data object
        graph = Data()
        for key in td.keys():
            if key == "data":
                pass
            else:
                try:
                    graph[key] = td[key].clone()
                except:
                    graph[key] = td[key]
        td["data"] = NonTensorData(graph)
        return td


class EWMARewardNormalizer:
    def __init__(self, alpha=0.01, epsilon=1e-8):
        self.mean = 0
        self.var = 1
        self.alpha = alpha  # Smaller alpha = slower adaptation
        self.epsilon = epsilon
        self.initialized = False

    def normalize(self, reward):
        # Extract scalar value if needed
        if isinstance(reward, torch.Tensor):
            reward_val = reward.item()
        else:
            reward_val = reward

        # Initialize values on first call
        if not self.initialized:
            self.mean = reward_val
            self.initialized = True
            return 0.0

        # Update running mean
        delta = reward_val - self.mean
        self.mean = self.mean + self.alpha * delta

        # Update running variance
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        std = math.sqrt(self.var) + self.epsilon

        # Return normalized reward
        return (reward_val - self.mean) / std



# class NetworkRunner(EnvBase): # Modeled after torhrl EnvBase
#     def __init__(self, env: EnvBase, type: str,  max_steps: int = 1000, actor = None, reward = "negative_backlog", alpha = 0.1, **kwargs):
#         super().__init__()
#         self.env = env
#         self.type = type
#         self.max_steps = max_steps
#         self.actor = actor
#         self.create_spec()
#
#         # reward normalization parameters
#         self.alpha = alpha
#
#         # if negative_backlog, then self.reward = -backlog
#         self.reward = reward
#
#         self.reward_normalizer = EWMARewardNormalizer(alpha=alpha)
#
#
#     def create_spec(self):
#
#         self.observation_spec = Composite({
#             "data": NonTensor(shape = (1,), dtype = torch.float32),
#             "Qavg": Unbounded(shape =-1),
#         })
#         self.action_spec = Composite({
#             "bias": Unbounded(shape =-1)
#         })
#
#
#     def _reset(self, *args, **kwargs):
#         return self.env.get_rep()
#
#     def _set_seed(self, seed: Optional[int] = None):
#         return
#
#     def _step(self, td: TensorDict = None) -> TensorDict:
#         return self.get_run(td)
#
#     def get_run(self, tensordict):
#
#         if "data" in tensordict:
#             Warning(" `data` already in td")
#
#         if "W" in tensordict:
#             if self.type == "bp":
#                 self.env.set_bias(tensordict["W"])
#             elif self.type == "randomized":
#                 self.env.W = tensordict["W"].squeeze()
#
#         # initialize tensordict
#         td = self.env.get_rep()
#
#         # collect rollout data
#         rollout = self.env.rollout(max_steps = self.max_steps)
#
#         # record metrics
#         td["backlog"] = rollout["backlog"].mean()
#
#         # get the routing coefficients
#         td["W"] = self.env.bias.clone()
#
#         # calculate reward
#         # calculate reward
#         reward_baseline_ratio = 1
#         if self.reward == "negative_backlog":
#             reward = -td["backlog"].mean()
#             if "SPBP" in self.env.baselines.keys():
#                 reward_baseline_ratio = reward/self.env.baselines["SPBP"]["backlog"]
#         elif self.reward == "negative_power_backlog":
#             reward = -(td["power"] + td["backlog"]).mean()
#         elif self.reward == "worst_class_backlog":
#             pass
#
#
#
#         td["reward"] = reward
#         td["reward_baseline_ratio"] = reward_baseline_ratio
#         td["reward_baseline_ratio"] = reward/self.baseline_reward
#         td["norm_reward"] = self.reward_normalizer.normalize(reward)
#         td["link_rewards"] = td["norm_reward"] * torch.ones_like(td["W"])
#
#         td["num_steps"] = len(rollout)
#
#         # get data from tensordict
#         if tensordict is not None:
#             for key in tensordict.keys():
#                 if key in ROUTING_KEYS:
#                     td[key] = tensordict[key]
#
#         td["context_id"] = self.env.context_id
#
#         # Create Data object
#         graph = Data()
#         for key in td.keys():
#             if key == "data":
#                 pass
#             else:
#                 try:
#                     graph[key] = td[key].clone()
#                 except:
#                     graph[key] = td[key]
#         td["data"] = NonTensorData(graph)
#         return td



if __name__ == "__main__":

    file_path = "../envs/grid_3x3.json"
    env_info = json.load(open(file_path, 'r'))
    env_info["action_func"] = "bpi"
    runner = create_network_runner(MultiClassMultiHopBP(**env_info), max_steps = 1000, graph = True)
    td = runner._step()

