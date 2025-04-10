
# import
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import networkx as nx

import numpy as np
import torch
from tensordict import TensorDict, merge_tensordicts

from torchrl.data import Composite, Bounded, Unbounded, Binary

from torchrl.envs import (
    EnvBase,
)

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FakeRV:

    def __init__(self, val = 1):
        self.val = 1

    def sample(self):
        return self.val

    def mean(self):
        return self.val


def poisson_sampler(rates, generator):
    return torch.poisson(rates, generator=generator)
def fixed_sampler(rates):
    return rates


class MultiClassMultiHopTS(EnvBase):
    """
    Environment for Multi-Class Multi-Hop routing with link buffers.

    Internal state:
    - Q (Tensor[NxK]): Per-node/class arrival queues
    - L (List[Queue]): FIFO queues for each link (all classes buffered together)
    - L_classes (List[List]): Class information for each packet in the link queue
    - Y (Tensor[Mx1]): Number of packets that can be sent on each link
    - W (Tensor[MxK]): Weights for each link-class pair

    Actions:
    - Set of weights for each node's outgoing link-class pairs
    """

    batch_locked = False

    def __init__(self,
                 nodes: Union[List[int], int],
                 link_info: List[Dict],
                 class_info: Dict,
                 link_distribution: str = "fixed",
                 arrival_distribution: str = "poisson",
                 single_path: bool = False,
                 context_id: Optional[int] = 0,
                 max_backlog: Optional[int] = None,
                 device: Optional[str] = None,
                 seed: int = 0,
                 **kwargs):
        super().__init__()

        # Set device
        self.device = device

        # Set batch size
        self.batch_size = torch.Size()

        # Context ID
        self.context_id = torch.tensor(context_id, dtype=torch.float32)

        # Single path routing
        self.single_path = single_path

        # Maximum backlog
        self.max_backlog = max_backlog

        # Baselines for performance comparison
        self.baselines = kwargs.get("baselines", {})

        # Random number generator
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

        # Nodes setup
        self.nodes = sorted(nodes) if isinstance(nodes, list) else list(range(nodes))
        self.num_nodes = self.N = len(self.nodes)

        # Links setup
        self.link_info = link_info
        self.num_links = self.M = len(link_info)
        self.link_distribution_type = link_distribution.lower()
        self.link_map = {}
        self.service_rates = torch.zeros(self.M)
        self.link_power = torch.zeros(self.M)
        self.outgoing_links = OrderedDict()
        self.incoming_links = OrderedDict()

        self._init_links(link_info)
        self.edge_index = torch.transpose(torch.tensor(list(self.link_map.values()), dtype=torch.long), 0, 1)
        self.link_rates = torch.tensor([link_dict["rate"] for link_dict in link_info], dtype=torch.float32)
        self.cap = self._sim_capacities()
        self.start_nodes = torch.tensor([self.link_map[m][0] for m in range(self.M)])
        self.end_nodes = torch.tensor([self.link_map[m][1] for m in range(self.M)])

        # Classes setup
        self.class_info = class_info = {int(k): {int(kk): v for kk, v in v.items()} for k, v in class_info.items()}
        self.num_classes = self.K = len(class_info)

        # Initialize node queues for each class
        self.Q = torch.zeros([self.N, self.K])

        # Initialize link queues (as lists for FIFO behavior)
        self.L = [[] for _ in range(self.M)]
        self.L_classes = [[] for _ in range(self.M)]
        self.L_class_count = torch.zeros([self.M, self.K], dtype=torch.int32)

        # Initialize weights
        self.W = torch.ones([self.M, self.K]) / self.M

        self.arrival_rates = torch.zeros((self.N, self.N), dtype=torch.float32)
        self.class_arrival_rates = torch.zeros([self.N, self.K], dtype=torch.float32)
        self.class_arrival_map = {}
        self.destination_features = torch.zeros([self.N, self.K], dtype=torch.float32)
        self.destination_map = {}
        self.class_map = {}
        self.arrival_distribution_type = arrival_distribution.lower()

        self._init_classes(class_info)

        # Initialize Graph
        self.graphx = nx.DiGraph()
        self.graphx.add_weighted_edges_from(
            [(self.link_map[m][0], self.link_map[m][1], 1 / self.link_rates[m]) for m in range(self.M)])

        self.max_shortest_path = 100
        self.shortest_path_dist = {}
        self.weighted_shortest_path_dist = {}
        self.sp_dist = torch.zeros([self.N, self.K])
        self.weighted_sp_dist = torch.zeros([self.N, self.K])

        self._init_shortest_path_dist()
        self._init_weighted_shortest_path_dist()

        self.base_mask = torch.zeros([self.M, self.K], dtype=torch.bool)
        self._init_base_mask()

        # Create specs for torchrl
        self._make_spec()

    def _make_spec(self):
        """Initialize specs for TorchRL compatibility"""
        self.observation_spec = Composite({
            "Q": Unbounded(shape=self.Q.shape, dtype=torch.float32),
            "L_sizes": Unbounded(shape=torch.Size([self.M]), dtype=torch.int32),
            "L_class_count": Unbounded(shape=self.L_class_count.shape, dtype=torch.int32),
            "backlog": Unbounded(shape=torch.Size([1]), dtype=torch.float32),
            "cap": Bounded(low=0, high=100, shape=self.cap.shape, dtype=torch.float32),
            "edge_index": Unbounded(shape=self.edge_index.shape, dtype=torch.long),
            "arrival_rates": Unbounded(shape=self.arrival_rates.shape, dtype=torch.float32),
            "link_rates": Unbounded(shape=self.link_rates.shape, dtype=torch.float32),
            "context_id": Unbounded(shape=self.context_id.shape, dtype=torch.float32),
            "mask": Binary(shape=(self.base_mask.shape[0], self.base_mask.shape[1] + 1), dtype=torch.bool),
            "departures": Unbounded(shape=self.Q.shape, dtype=torch.float32),
            "arrivals": Unbounded(shape=self.arrival_rates.shape, dtype=torch.float32),
            "sp_dist": Unbounded(shape=self.sp_dist.shape, dtype=torch.float32),
            "W": Unbounded(shape=self.W.shape, dtype=torch.float32),
        }, batch_size=self.batch_size)

        # Action is setting weights for each link-class pair
        self.action_spec = Bounded(low=0, high=1, shape=torch.Size([self.M, self.K]), dtype=torch.float32)
        self.reward_spec = Unbounded(shape=torch.Size([1]), dtype=torch.float32)

    def _set_seed(self, seed):
        """Set random seed"""
        self.rng.manual_seed(seed)
        self.seed = seed

    def _init_links(self, link_info):
        """Initialize link information"""
        for id, link_dict in enumerate(link_info):
            self.link_map[id] = (link_dict["start"], link_dict["end"])
            self.service_rates[id] = link_dict["rate"]
            self.link_power[id] = link_dict.get("power", 0)
            self.outgoing_links[link_dict["start"]] = self.outgoing_links.get(link_dict["start"], []) + [id]
            self.incoming_links[link_dict["end"]] = self.incoming_links.get(link_dict["end"], []) + [id]

    def _sim_capacities(self):
        """Simulate link capacities based on distribution"""
        if self.link_distribution_type == "fixed":
            return self.link_rates
        elif self.link_distribution_type == "poisson":
            return torch.poisson(self.service_rates, generator=self.rng)
        else:
            raise ValueError("Distribution not supported")

    def _sim_arrivals(self):
        """Simulate arrivals based on distribution"""
        if self.arrival_distribution_type == "poisson":
            return torch.poisson(self.arrival_rates, generator=self.rng)
        elif self.arrival_distribution_type == "fixed":
            return self.arrival_rates
        else:
            raise ValueError("Distribution not supported")

    def _process_arrivals(self):
        """Process new packet arrivals into node queues"""
        arrivals = self._sim_arrivals()
        # Convert from NxN matrix to NxK matrix of arrivals
        arrivals_k = torch.zeros_like(self.Q)
        for destination, cls in self.class_map.items():
            arrivals_k[:, cls] = arrivals[:, destination]
        self.Q += arrivals_k
        return arrivals

    def _process_departures(self):
        """Process packets that have reached their destination"""
        departures = torch.zeros_like(self.Q)
        for id, dest in self.destination_map.items():
            departures[dest, id] = self.Q[dest, id].item()
            self.Q[dest, id] = 0
        return departures

    def _init_classes(self, class_info):
        """Initialize class information"""
        for id, destination in enumerate(list(class_info.keys())):
            self.destination_map[id] = int(destination)
            self.class_map[destination] = id
            self.class_arrival_map[id] = []
            for source, rate in class_info[destination].items():
                self.arrival_rates[int(source), int(destination)] = rate
                self.class_arrival_rates[int(source), id] = rate
                self.destination_features[int(destination), id] = 1
                self.class_arrival_map[id].append((int(source), rate))

    def _reset(self, tensordict=None, **kwargs):
        """Reset the environment"""
        # Reset node queues
        self.Q = torch.zeros_like(self.Q)

        # Reset link queues
        self.L = [[] for _ in range(self.M)]
        self.L_classes = [[] for _ in range(self.M)]

        # Reset weights
        # self.W = torch.ones([self.M, self.K]) / self.M

        # Generate capacities
        self.cap = self._sim_capacities()

        return self._get_observation(reset=True)

    def _step(self, td: TensorDict):
        """Execute environment step"""
        # Get action (weights) from the tensordict
        W = td.get("W")
        # Update the weights based on the action
        if W is not None and W.shape == self.W.shape:
            self.W = W
        else:
            raise ValueError("Invalid action shape")
        # Execute the step logic with the new weights
        return self._get_observation(action=W)

    def _get_observation(self, action=None, reset=False):
        """Get observation after a step or reset"""


        # STEP 1: Process packets from node queues to link queues based on weights
        self._route_packets_to_links()

        # STEP 2: Transfer packets from link queues to destination nodes
        transferred_packets = self._transfer_packets_from_links()

        # STEP 3: Deliver packets that have reached their destination
        departures = self._process_departures()

        # STEP 4: Generate new arrivals
        arrivals = self._process_arrivals()

        # STEP 5: Compute reward (negative of total backlog)
        total_backlog = self.Q.sum() + sum(len(queue) for queue in self.L)
        reward = -total_backlog.reshape([1])
        backlog = total_backlog.reshape([1])

        # STEP 6: Check if episode should be truncated
        terminated = False
        if self.max_backlog is not None and -reward > self.max_backlog:
            terminated = True

        # Get Power
        power = 0 #torch.sum(self.link_power * (torch.sum(transferred_packets, dim = 1) > 0))


        # STEP 7: Generate new link capacities
        self.cap = self._sim_capacities()

        # STEP 8: Generate mask
        mask = self._get_mask()

        # Create link queue size tensor for observation
        L_sizes = torch.tensor([len(queue) for queue in self.L], dtype=torch.int32)

        # Prepare the observation tensordict
        td = TensorDict({
            "Q": self.Q.clone(),
            "power": power,
            "L_sizes": L_sizes,
            "L_class_count": self.L_class_count.clone(),
            "backlog": backlog,
            "cap": self.cap.clone(),
            "departures": departures,
            "arrivals": arrivals,
            "mask": mask.clone(),
            "sp_dist": self.sp_dist,
            "edge_index": self.edge_index,
            "link_rates": self.link_rates,
            "arrival_rates": self.arrival_rates,
            "context_id": self.context_id,
            "terminated": torch.tensor([terminated], dtype=bool),
            "W": self.W.clone(),
        }, batch_size=self.batch_size)

        if action is not None:
            td.set("action", action)

        if not reset:
            td.set("reward", reward)

        return td

    def _route_packets_to_links(self):
        """Route packets from node queues to link queues based on weights"""
        # For each node and class
        for n in range(self.N):
            for k in range(self.K):
                if self.Q[n, k] <= 0:
                    continue

                # Get outgoing links for this node
                if n not in self.outgoing_links:
                    continue

                links = self.outgoing_links[n]
                if not links:
                    continue

                # Normalize weights for this node-class pair
                node_class_weights = self.W[links, k]

                # Mask invalid links
                node_class_weights = node_class_weights * self.base_mask[links, k]

                # If single path routing, only select the link with the highest weight
                if self.single_path:
                    max_weight = torch.max(node_class_weights)
                    max_indices = torch.where(node_class_weights == max_weight)[0]
                    selected_index = max_indices[0]  # Select the first occurrence
                    node_class_weights = torch.zeros_like(node_class_weights)
                    node_class_weights[selected_index] = max_weight

                total_weight = node_class_weights.sum()
                if total_weight > 0:
                    node_class_weights = node_class_weights / total_weight
                else:
                    node_class_weights = torch.ones_like(node_class_weights) / len(links)

                # Calculate packets to route to each link
                packets_to_route = self.Q[n, k].item()
                packets_per_link = torch.floor(node_class_weights * packets_to_route)
                remaining = packets_to_route - packets_per_link.sum().item()

                # Distribute remaining packets
                if remaining > 0:
                    _, indices = torch.sort(node_class_weights, descending=True)
                    for i in range(int(remaining)):
                        packets_per_link[indices[i % len(indices)]] += 1

                # Add packets to link queues
                for i, link in enumerate(links):
                    num_packets = int(packets_per_link[i].item())
                    if num_packets <= 0:
                        continue

                    # Add packets to link queue
                    self.L[link].extend([1] * num_packets)
                    self.L_classes[link].extend([k] * num_packets)
                    self.L_class_count[link, k] += num_packets


                    # Reduce packets from node queue
                    self.Q[n, k] -= num_packets

    def _transfer_packets_from_links(self):
        """Transfer packets from link queues to destination nodes based on link capacities"""
        # track the number of packets transferred on each link
        transferred_packets = torch.zeros(self.M, dtype=torch.int32)

        for m in range(self.M):
            if not self.L[m]:
                continue

            # Get link capacity (how many packets can be transferred)
            capacity = int(self.cap[m].item())
            if capacity <= 0:
                continue

            # Limit by queue size
            capacity = min(capacity, len(self.L[m]))

            # Transfer packets
            for _ in range(capacity):
                if not self.L[m]:
                    break

                # Remove packet from link queue (FIFO)
                _ = self.L[m].pop(0)
                class_id = self.L_classes[m].pop(0)
                self.L_class_count[m, class_id] -= 1


                # Add to destination node queue
                dest_node = self.end_nodes[m].item()
                self.Q[dest_node, class_id] += 1

                transferred_packets[m] += 1
        return transferred_packets


    def _init_shortest_path_dist(self):
        """Initialize shortest path distances"""
        for source, dest_dict in nx.all_pairs_shortest_path_length(self.graphx):
            self.shortest_path_dist[source] = {}
            for n in range(self.N):
                if n in dest_dict:
                    self.shortest_path_dist[source][n] = dest_dict[n]
                else:
                    self.shortest_path_dist[source][n] = self.max_shortest_path

        for n in range(self.N):
            for k in range(self.K):
                try:
                    self.sp_dist[n, k] = self.shortest_path_dist[n][self.destination_map[k]]
                except KeyError:
                    self.sp_dist[n, k] = self.max_shortest_path
        return self.sp_dist

    def _init_weighted_shortest_path_dist(self):
        """Initialize weighted shortest path distances"""
        for source, dest_dict in nx.all_pairs_dijkstra_path_length(self.graphx):
            self.weighted_shortest_path_dist[source] = {}
            for n in range(self.N):
                if n in dest_dict:
                    self.weighted_shortest_path_dist[source][n] = dest_dict[n]
                else:
                    self.weighted_shortest_path_dist[source][n] = self.max_shortest_path

        for n in range(self.N):
            for k in range(self.K):
                try:
                    self.weighted_sp_dist[n, k] = self.weighted_shortest_path_dist[n][self.destination_map[k]]
                except KeyError:
                    self.weighted_sp_dist[n, k] = self.max_shortest_path

    def _init_base_mask(self):
        """Initialize base mask for valid actions"""
        for m in range(self.M):
            for k in range(self.K):
                # Check if end node of link m can reach destination of class k
                self.base_mask[m, k] = self.shortest_path_dist[self.end_nodes[m].item()][
                                           self.destination_map[k]] < self.max_shortest_path

    def _get_mask(self):
        """Get mask for valid actions"""
        # Can only route packets to links than can reach destination
        return self.base_mask

    def get_net_spec(self, include=["sp_dist"]):
        """Get network specification"""
        return self.get_rep(include=include)

    def get_rep(self, features=("link_rates", "destination", "arrival_rates")) -> TensorDict:
        """
        Returns the representation of the network needed for the GNN
        Node features (X) descriptions:
            destination_features (default arg): For each node, 1xK tensor where the kth element is 1 if the destination of class k is the node
            sp_dist: For each node, 1xK tensor where the kth element is the shortest path distance from the node to the destination of class k
            class_arrival_rates (always included): For each node, 1xK tensor where the kth element is the arrival rate of class k at the node

        Edge features (edge_attr) descriptions:
            link_rates (always included): For each link, 1xK tensor where the kth element is the average rate of class k packets on the link (should be the same for all classes)
            bias: For each link, 1xK tensor where the kth element is the shortest bias for class k packets on the link (should not really be used)
            link_power: For each link, 1xK tensor where the kth element is the power of class k packets on the link (should be the same for all classes)

        :return: td
        """
        potential_edge_features = ["bias", "link_rates", "link_power"]
        potential_node_features = ["destination", "sp_dist", "arrival_rates"]

        edge_features = []
        node_features = []
        for feat in features:
            if feat in potential_edge_features:
                if feat == "link_rates":
                    link_rates = self.link_rates.unsqueeze(-1).expand(-1, self.K)
                    edge_features.append(link_rates)
                elif feat == "link_power":
                    link_power = self.link_power.unsqueeze(-1).expand(-1, self.K)
                    edge_features.append(link_power)
                elif feat == "bias":
                    edge_features.append(self.bias)
            elif feat in potential_node_features:
                if feat == "destination":
                    node_features.append(self.destination_features)
                elif feat == "sp_dist":
                    node_features.append(self.sp_dist)
                elif feat == "arrival_rates":
                    node_features.append(self.class_arrival_rates)
            else:
                raise ValueError(f"Feature {feat} not supported")

        edge_attr = torch.stack(edge_features, dim=2)
        X = torch.stack(node_features, dim=2)

        # if "bias" in include:
        #     # expand link rates to shape of bias
        #     link_rates = self.link_rates.unsqueeze(-1).expand(-1, self.K)
        #     edge_attr = torch.stack([link_rates, self.bias], dim = 2)
        #     # raise NotImplementedError("Include bias not yet implemented")
        # elif "link_power" in include:
        #     link_rates = self.link_rates.unsqueeze(-1).expand(-1, self.K)
        #     link_power = self.link_power.unsqueeze(-1).expand(-1, self.K)
        #     edge_attr = torch.stack([link_rates, link_power], dim = 2)
        # else:
        #     edge_attr = self.link_rates.unsqueeze(-1).expand(-1, self.K).unsqueeze(-1)
        # if "destination" in include:
        #     X = torch.stack((self.class_arrival_rates, self.destination_features), dim = 2)
        # elif "sp_dist" in include:
        #     X = torch.stack((self.class_arrival_rates, self.sp_dist), dim = 2)
        # else:
        #     X = self.class_arrival_rates.unsqueeze(-1)
        # # TODO: Onehot encode destination map and add to node feature

        return TensorDict({"X": X,
                           "edge_attr": edge_attr,
                           "edge_index": self.edge_index}
                          )

    # def get_rep(self, include=["destination_features"]) -> TensorDict:
    #     """Get representation of the network"""
    #     if "bias" in include:
    #         link_rates = self.link_rates.unsqueeze(-1).expand(-1, self.K)
    #         edge_attr = torch.stack([link_rates, torch.zeros_like(link_rates)], dim=2)
    #     else:
    #         edge_attr = self.link_rates.unsqueeze(-1).expand(-1, self.K).unsqueeze(-1)
    #
    #     if "destination_features" in include:
    #         X = torch.stack((self.class_arrival_rates, self.destination_features), dim=2)
    #     elif "sp_dist" in include:
    #         X = torch.stack((self.class_arrival_rates, self.sp_dist), dim=2)
    #     else:
    #         X = self.class_arrival_rates.unsqueeze(-1)
    #
    #     return TensorDict({
    #         "X": X,
    #         "edge_attr": edge_attr,
    #         "edge_index": self.edge_index
    #     })

    # def get_L_class_count(self):
    #     """Count the number of packets in each link queue for each class"""
    #     return torch.tensor([[self.L_classes[m].count(k) for k in range(self.K)] for m in range(self.M)])




def create_sp_bias(net,weighted = False,  alpha=1):
    """
    Given a network instance get the shortest path bias for each link and class,
    where the shortest path bias of a link, class pair is the shortest path distance from the end node
    to the destination of said class
    :param net:
    :return:
    """
    if weighted:
        node_sp_dist = net.weighted_sp_dist if hasattr(net, "weighted_sp_dist") else net._init_weighted_shortest_path_dist()
    else:
        node_sp_dist = net.sp_dist if hasattr(net, "sp_dist") else net._init_shortest_path_dist()
    link_sp_dist = node_sp_dist[net.start_nodes]- node_sp_dist[net.end_nodes]
    bias = alpha/link_sp_dist
    bias[torch.isinf(bias)] = 0
    bias[torch.isnan(bias)] = 0
    return bias, link_sp_dist


# def create_link_weights(env, method = "capacity_weighted_shortest_path"):
#     """
#     The weight of link (i,j) for class k is the difference between the shortest path difference
#     between node i and node j to the destination of class k, where link distance is equal to
#     the inverse of link capacity
#
#     :param env:
#     :param method:
#     :return:
#     """
#     METHODS = ["capacity_weighted_shortest_path", "shortest_path", "uniform"]
#     if method not in METHODS:
#         raise ValueError(f"Method {method} not supported. Supported methods are {METHODS}")
#     if method == "capacity_weighted_shortest_path":
#         weights, link_sp_dist = create_sp_bias(env,weighted = True)
#     elif method == "shortest_path":
#         weights, link_sp_dist = create_sp_bias(env,weighted = False)
#
#     return weights.clip(0)


def create_link_weights(env, method = "capacity_weighted_shortest_path"):
    """
    The weight of link (i,j) for class k is the difference between the shortest path difference
    between node i and node j to the destination of class k, where link distance is equal to
    the inverse of link capacity

    :param env:
    :param method:
    :return:
    """
    METHODS = ["capacity_weighted_shortest_path", "shortest_path", "inverse_capacity_weighted_shortest_path"]
    if method not in METHODS:
        raise ValueError(f"Method {method} not supported. Supported methods are {METHODS}")
    if method == "capacity_weighted_shortest_path":
        """
        w_{i,j} = (node_sp_dist[j,k] + 1/link_rate[(i,j)]) if (node_sp_dist[i,k] - node_sp_dist[j,k] >= 0 else 0
        
        Prevents routing packets to nodes that are "further away" from the destination than the current node 
        """
        node_sp_dist = env.weighted_sp_dist if hasattr(env, "weighted_sp_dist") else env._init_weighted_shortest_path_dist()
        link_sp_dist_diff = node_sp_dist[env.start_nodes]- node_sp_dist[env.end_nodes]
        weights = (link_sp_dist_diff >= 0).float()/(node_sp_dist[env.end_nodes] + (env.link_rates).unsqueeze(1).expand(-1, env.K))
    elif method == "shortest_path":
        node_sp_dist = env.sp_dist if hasattr(env, "sp_dist") else env._init_shortest_path_dist()
        link_sp_dist_diff = node_sp_dist[env.start_nodes]- node_sp_dist[env.end_nodes]
        weights = (link_sp_dist_diff >= 0).float()/(node_sp_dist[env.end_nodes] + 1)
    elif method == "inverse_capacity_weighted_shortest_path":
        node_sp_dist = env.weighted_sp_dist if hasattr(env, "weighted_sp_dist") else env._init_weighted_shortest_path_dist()
        link_sp_dist_diff = node_sp_dist[env.start_nodes]- node_sp_dist[env.end_nodes]
        weights = (link_sp_dist_diff >= 0).float()/(node_sp_dist[env.end_nodes]+ 1/env.link_rates.unsqueeze(1).expand(-1, env.K))

    return weights.clip(0)







