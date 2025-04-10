import os
from copy import deepcopy
import numpy as np
from typing import List



from .MultiClassMultihopBP import MultiClassMultiHopBP
from .MultiClassMultihopTS import MultiClassMultiHopTS


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def make_env_mcmh_bp(
        env_params,
        seed: int = 0,
        max_backlog = None,
        *args, **kwargs
        ):
    env_params = deepcopy(env_params)
    env_params["seed"] = seed
    env_params["max_backlog"] = max_backlog
    for keyword in kwargs.keys():
        env_params[keyword] = kwargs[keyword]

    env = MultiClassMultiHopBP(**env_params)

    return env


def make_env_mcmh_ts(
        env_params,
        seed: int = 0,
        max_backlog = None,
        *args, **kwargs
        ):

    env_params = deepcopy(env_params)
    env_params["seed"] = seed
    env_params["max_backlog"] = max_backlog
    for keyword in kwargs.keys():
        env_params[keyword] = kwargs[keyword]
    env = MultiClassMultiHopTS(**env_params)
    return env



class EnvSampler:
    """
    Takes in either a single environment parameter file and returns an instance of the environment on sample()
    or takes in a generated env parameter file which contains many different instances of the environment
    """

    def __init__(self, input_params,
                 make_env_keywords = None,
                 env_generator_seed = 0,
                 cycle_sample = False,
                 make_env_fn = make_env_mcmh_bp,
                 *args, **kwargs):

        self.make_env_fn = make_env_fn

        # if env_params has a key "key_params" then is the parameters of many environments
        self.context_dicts = None

        # set np seed
        self.env_generator_seed = env_generator_seed
        self.seed_generator = np.random.default_rng(env_generator_seed)

        # check if env_params is a single environment or many
        if "num_envs" in input_params.keys():
            self.context_dicts = input_params["context_dicts"]
            # if all keys of context_dicts are str, then convert to int
            if all([isinstance(key, str) for key in self.context_dicts.keys()]):
                self.context_dicts = {int(key): value for key, value in self.context_dicts.items()}
            self.num_envs = input_params["num_envs"]
            if not cycle_sample:
                self.sample = self.sample_from_multi
            else:
                self.sample = self.cycle_sample
                self.last_sample_ind = -1
        # if all entries are dicts, then each dict is its
        elif all([isinstance(value, dict) for value in input_params.values()]):
            self.context_dicts = input_params
            if all([isinstance(key, str) for key in self.context_dicts.keys()]):
                self.context_dicts = {int(key): value for key, value in self.context_dicts.items()}
            self.num_envs = len(input_params.keys())
            if not cycle_sample:
                self.sample = self.sample_from_multi
            else:
                self.sample = self.cycle_sample
                self.last_sample_ind = -1
        else:
            self.context_dicts = {0: {
                                    "env_params":input_params,
                                   "admissible": None,
                                   "arrival_rates": input_params.get("arrival_rates", None),
                                   "lta": input_params.get("lta", None),
                                   "network_load": input_params.get("network_load", None)}}
            self.num_envs = 1
            self.sample = self.sample_from_multi
        self._make_env_keywords = make_env_keywords
        self.history = []

    def clear_history(self):
        self.history = []

    def reseed(self, seed = None):
        if seed is None:
            seed = self.env_generator_seed
        self.seed_generator = np.random.default_rng(seed)
        self.env_generator_seed = seed

    def gen_seeds(self, n):
        return self.seed_generator.integers(low = 0, high = 100000, size = n)


    def sample_from_multi(self, index = None, seed = None):
        """
        If given rel_ind, then samples the rel_ind-th environment from the context_dicts
        If given true_ind, then samples the true_ind environment from the context_dicts
        If neither are given, then samples from the context_dicts with a random index uniformly
        :param rel_ind:
        :param true_ind:
        :return:
        """

        if index is None:

            index = self.seed_generator.choice(list(self.context_dicts.keys()))

        try:
            env_params = self.context_dicts[index]["env_params"]
        except KeyError:
            env_params = self.context_dicts[index]
            env_params["context_id"] = index
            # env_params["bp_lta"] = self.context_dicts[index].get("bp_lta", None)
            # env_params["spbp_lta"] = self.context_dicts[index].get("spbp_lta", None)
        # if ind is not in the keys
        except KeyError:
            raise ValueError(f"Index {index} is not in the keys of the environment parameters")
        if seed is None:
            seed = self.seed_generator.integers(low = 0, high = 100000)
        env = self.make_env_fn(env_params, seed = seed, **self._make_env_keywords)
        self.history.append(index)
        return env

    def cycle_sample(self):
        """On each call, the next environment is sampled"""
        self.last_sample_ind += 1
        if self.last_sample_ind >= self.num_envs:
            self.last_sample_ind = 0
        return self.sample_from_multi(self.last_sample_ind)

    def sample_from_solo(self):
        env = self.make_env_fn(self.context_dicts, seed = self.seed_generator.integers(low = 0, high = 100000), **self._make_env_keywords)
        env.baseline_lta = self.baseline_lta
        self.history.append(0)
        return env


    def create_all_envs(self):
        envs = {}
        for i in range(self.num_envs):
            key = list(self.context_dicts.keys())[i]
            env_params = self.context_dicts[key]
            if "env_params" in env_params.keys():
                env_params = env_params["env_params"]
            env = self.make_env_fn(env_params, **self._make_env_keywords)
            envs[i] = {"env":env,
                      "env_params":env_params,
                      "ind": i,
                      "arrival_rates":key}
        return envs

    def add_env_params(self, env_params, ind = None):
        if ind is None:
            ind = self.num_envs
        self.context_dicts[ind] = env_params
        self.num_envs += 1



