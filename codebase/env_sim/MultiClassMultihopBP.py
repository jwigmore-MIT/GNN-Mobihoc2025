
# import
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import networkx as nx

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


class MultiClassMultiHopBP(EnvBase): #
    """
    Internal state is Q and Y (queues and link states)
    Q is size NxK where N is the number of nodes in the network and K is the number of classes
    Y is size Mx1 where M is the number of links in the network

    Actions are Mx(K+1) matrices where M is the number of links and K is the number of classes
    Action[m,k+1] denotes the number of packets to send on link m for class k
    Action[m,0] denotes the number of packets to idle on link m



    Matrix Representation:
        Y is a NxN matrix
        Q is a NxK matrix
        Action is a NxNx(K+1)
        Action[n,m,k+1] denotes the number of packets to send from node n to node m for class k

        Dynamics: Q[n,k] = max(Q[n,k] - sum(Action[n,:,k+1]), 0) + sum(Action[:,n,k+1])





    The reward is the negative of the sum of the elements of Q at time t, meaning before the action and new arrivals

    X(t) is sampled from a set of arrival distributions, which is a list of N+1 distributions (generators)
    Y(t) is sampled from a set of link state distributions, which is a list of N+1 distributions (generators)

    Notes:
        1. Assuming batch_size = 1 (i.e.number of environments each instance represents is 1)
        2.


    """



    batch_locked = False
    def __init__(self,
                 nodes: Union[List[int], int],
                 link_info: Dict,
                 class_info: Dict,
                 link_distribution: str = "fixed",
                 arrival_distribution: str = "poisson",
                 context_id: Optional[int] = 0,
                 max_backlog: Optional[int] = None,
                 device: Optional[str] = None,
                 action_func: Optional[str] = None,
                 bias = None,
                 power_penalty = 0,
                 seed = 0,
                 **kwargs):
        super().__init__()

        # Set device -  where incoming and outgoing tensors lie
        self.device = device

        # Set Batch Size
        self.batch_size = torch.Size()

        # Interval ID
        self.context_id = torch.tensor(context_id, dtype = torch.float32)

        # Maximum backlog
        self.max_backlog = max_backlog

        # Baseline performance
        self.baselines = kwargs.get("baselines", {})

        # Power penalty
        self.power_penalty = power_penalty

        # Internal random number generator
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

        # set action function
        if action_func == "bpi":
            self.action_func = self.backpressureWithInterference
        else:
            self.action_func = self.backpressure


        # Nodes/Buffers/Queues
        self.nodes = sorted(nodes) if isinstance(nodes, list) else list(range(nodes))
        self.num_nodes = self.N = len(self.nodes)

        # Initialize Links
        self.link_info = link_info
        self.num_links = self.M = len(link_info)
        self.link_distribution_type = link_distribution.lower()
        self.link_map = {}  #
        self.service_rates = torch.zeros(self.M)
        self.link_power = torch.zeros(self.M)
        self.outgoing_links = OrderedDict() # for each node, a list of outgoing links
        self.incoming_links = OrderedDict()

        self._init_links(link_info)
        self.edge_index = torch.transpose(torch.tensor(list(self.link_map.values()), dtype = torch.long), 0, 1)
        self.link_rates = torch.tensor([link_dict["rate"] for link_dict in link_info], dtype = torch.float32)
        self.cap = self._sim_capacities() # Capacities at time $t=0$
        self.start_nodes = torch.tensor([self.link_map[m][0] for m in range(self.M)]) # Start node for each link
        self.end_nodes = torch.tensor([self.link_map[m][1] for m in range(self.M)])   # End node for each link

        # Initialize Classes
        # convert class info keys and nested keys to integers
        self.class_info = class_info = {int(k): {int(kk): v for kk, v in v.items()} for k, v in class_info.items()}
        self.num_classes = self.K = len(class_info)

        self.Q = torch.zeros([self.N, self.K])
        # self.arrival_map = {}  # Maps (start, end) to class id
        self.arrival_rates = torch.zeros((self.N,self.N), dtype=torch.float32)
        self.class_arrival_rates = torch.zeros([self.N, self.K], dtype = torch.float32)
        self.class_arrival_map = {} # Maps class to list of (source_node, rate) tuples
        self.destination_features = torch.zeros([self.N, self.K], dtype=torch.float32)
        self.destination_map = {} # Maps class id to destination node
        self.class_map = {} # Maps destination to class id
        self.arrival_distribution_type = arrival_distribution.lower()

        self._init_classes(class_info)
        # self._process_arrivals() # Creates initial arrivals at $t=0$

        # Initialize Bias
        if bias is None:
            self.bias = torch.zeros([self.M, self.K], dtype = torch.float32)
        else:
            self.bias = torch.tensor(bias, dtype = torch.float32) # Bias for each link and class
            if bias.shape != (self.M, self.K):
                raise ValueError(f"bias must have shape (M,K) ({self.M}, {self.K})")

        # Initialize Graph
        self.graphx = nx.DiGraph(list(self.link_map.values()))
        self.graphx = nx.DiGraph()
        # Set weight for each edge equal to the link rates
        self.graphx.add_weighted_edges_from([(self.link_map[m][0], self.link_map[m][1], 1/self.link_rates[m]) for m in range(self.M)])

        self.max_shortest_path = 100
        self.shortest_path_dist = {} # shortest path from node i to destination node of class k
        self.weighted_shortest_path_dist = {}
        self.sp_dist = torch.zeros([self.N, self.K]) #Tensor form of shortest_path_dist
        self.weighted_sp_dist = torch.zeros([self.N, self.K]) # Tensor form of shortest_path_dist with weights

        self._init_shortest_path_dist()
        self._init_weighted_shortest_path_dist()

        self.base_mask = torch.zeros([self.M, self.K], dtype = torch.bool)
        self._init_base_mask()

        # Create specs for torchrl
        self._make_spec()

    def _make_spec(self):
        """
        Ned to initialize the
        :return:
        """

        self.observation_spec = Composite({
            "Q": Unbounded(shape = self.Q.shape, dtype = torch.float32),
            "backlog": Unbounded(shape = torch.Size([1]), dtype = torch.float32),
            "cap": Bounded(low = 0, high = 100, shape = self.cap.shape, dtype = torch.float32),
            "edge_index": Unbounded(shape = self.edge_index.shape, dtype = torch.long),
            "arrival_rates": Unbounded(shape = self.arrival_rates.shape, dtype = torch.float32),
            "link_rates": Unbounded(shape = self.link_rates.shape, dtype = torch.float32),
            "context_id": Unbounded(shape = self.context_id.shape, dtype = torch.float32),
            "mask": Binary(shape = (self.base_mask.shape[0], self.base_mask.shape[1]+1), dtype = torch.bool),
            "departures": Unbounded(shape = self.Q.shape, dtype = torch.float32),
            "arrivals": Unbounded(shape = self.arrival_rates.shape, dtype = torch.float32),
            "sp_dist": Unbounded(shape = self.sp_dist.shape, dtype = torch.float32),


        }, batch_size=self.batch_size)
        self.action_spec = Bounded(low = 0, high = 1, shape = torch.Size([self.M, self.K+1]), dtype = torch.int)
        self.reward_spec = Unbounded(shape = torch.Size([1]), dtype = torch.float32)


    def _set_seed(self, seed):
        self.rng.manual_seed(seed)# all seeding should be done externally
        self.seed = seed
    def _get_link_cap(self, start, end):
        return self.cap[self.link_map[(start, end)]]

    def _init_links(self, link_info):
        for id,link_dict in enumerate(link_info):
            self.link_map[id] = (link_dict["start"], link_dict["end"])
            self.service_rates[id] = link_dict["rate"]
            self.link_power[id] = link_dict.get("power", 0)
            self.outgoing_links[link_dict["start"]] = self.outgoing_links.get(link_dict["start"], []) + [id]
            self.incoming_links[link_dict["end"]] = self.incoming_links.get(link_dict["end"], []) + [id]
        # self._sim_capacities =  self._init_random_process(self.service_rates, self.link_distribution_type, self.rng)

    def _sim_capacities(self):
        if self.link_distribution_type == "fixed":
            return self.link_rates
        elif self.link_distribution_type == "poisson":
            return torch.poisson(self.service_rates, generator=self.rng)
        else:
            raise ValueError("Distribution not supported")

    def _sim_arrivals(self):
        if self.arrival_distribution_type == "poisson":
            return torch.poisson(self.arrival_rates, generator=self.rng)
        elif self.arrival_distribution_type == "fixed":
            return self.arrival_rates
        else:
            raise ValueError("Distribution not supported")

    def _process_arrivals(self):
        arrivals = self._sim_arrivals() # NxN matrix of arrivals
        # need to get the N x K matrix of arrivals
        arrivals_k = torch.zeros_like(self.Q)
        for destination, cls in self.class_map.items():
            arrivals_k[:,cls] = arrivals[:,destination]
        self.Q += arrivals_k
        return arrivals

    def _process_departures(self):
        departures = torch.zeros_like(self.Q)
        for id, dest in self.destination_map.items():
            departures[dest, id] = self.Q[dest,id].item()
            self.Q[dest,id] = 0
        return departures

    def _init_classes(self, class_info):
        if isinstance(class_info, list):
            raise ValueError("class_info should be a dictionary (You are using the wrong MultiClassMultiHop class)")


        for id, destination in enumerate(list(class_info.keys())):
            # destination = int(destination)
            self.destination_map[id] = int(destination)
            self.class_map[destination] = id
            self.class_arrival_map[id] = []
            for source, rate in class_info[destination].items():
                # source = int(source)
                self.arrival_rates[int(source), int(destination)] = rate
                self.class_arrival_rates[int(source), id] = rate
                self.destination_features[int(destination), id] = 1
                self.class_arrival_map[id].append((int(source), rate))
        # self._sim_arrivals = self._init_random_process(self.arrival_rates, self.arrival_distribution_type, self.rng)




    # def _init_random_process(self, rates, dist, rng):
    #     """
    #     kwargs should contain the distribution (distribution = "poisson) and any parameters
    #     :param kwargs:
    #     :return:
    #     """
    #
    #     if not isinstance(rates,torch.Tensor):
    #         rates = torch.Tensor(rates)
    #     if dist == "poisson":
    #         return poisson_sampler(rates, rng)
    #     elif dist == "fixed":
    #         return fixed_sampler(rates)
    #     else:
    #         raise ValueError("Distribution not supported")
    def _reset(self, tensordict = None, **kwargs):

        self.Q = torch.zeros_like(self.Q)
        return self._get_observation(reset = True)

    def _step(self, td:TensorDict):
        action = self.action_func(self.Q, self._get_mask(),)
        return self._get_observation(action = action)


    def backpressure(self, *args, **kwargs):
        """
        Runs the backpressure algorithm given the network

        Backpressure algorithm:
            For each link:
                1. Find the class which has the greatest difference between the queue length at the start and end nodes THAT IS NOT MASKED
                2. Send the largest class using the full link capacity for each link if there is a positive differential between start and end nodes
                3. If there is no positive differential, send no packets i.e. a_i[0,1:] = 0, a_i[0,0] = Y[i]

        """

        Q = self.Q.clone()
        mask = self._get_mask()

        # Initialize action tensor
        action = torch.zeros([self.M, self.K])

        # Compute the pressure over each link for each class
        # Mask ensures that the pressure is 0 for classes that are not allowed to be sent
        pressure = (Q[self.start_nodes] - Q[self.end_nodes]  + self.bias) * mask[:, 1:]

        if self.power_penalty != 0:
            # broadcast the power penalty to the shape of pressure
            pressure = pressure - self.power_penalty * self.link_power

        # Find the class with the maximum pressure for each link
        weights, chosen_class = torch.max(pressure, dim=1, keepdim=True)

        # For each node transmit in order of the maxweight for each class
        # Apply Action
        start_Q = self.Q.clone()
        diffQ = torch.zeros_like(self.Q)

        for n in range(self.N):
            # get the links that start at node n
            if n not in self.outgoing_links:
                continue
            links = self.outgoing_links[n]
            link_weights = weights[links]
            cap = self.cap[links].unsqueeze(-1)
            # sort (links, link_weights) in descending order of link_weights
            links = [x for _, x in sorted(zip(link_weights*cap, links), reverse=True)]
            link_weights = sorted(link_weights, reverse=True)
            for link, weight in zip(links,link_weights):
                if weight <= 0: # backlog - power_cost + bias <= 0, meaning we don't send packets for this link (or any other outgoing links because weights are sorted)
                    break
                if start_Q[self.start_nodes[link], chosen_class[link]] == 0:
                    continue
                to_transmit = torch.min(self.cap[link], start_Q[self.start_nodes[link], chosen_class[link]])
                action[link, chosen_class[link]] = to_transmit
                start_Q[self.start_nodes[link], chosen_class[link]] -= to_transmit
                diffQ[self.start_nodes[link], chosen_class[link]] -= to_transmit
                diffQ[self.end_nodes[link], chosen_class[link]] += to_transmit

        self.Q += diffQ
        return action



    def backpressureWithInterference(self, *args, **kwargs):
        """
        Runs the backpressure algorithm with single transmission per node interference - meaning each node can activate
        a single outgoing link in each timestep. This allows scheduling to be done in a distributed per-node manner

        Can also incorporate power-delay tradeoff by adding a power penalty to the backpressure algorithm

        Can incorporate bias by defining a link-class bias for each link

        Algorithm:
        1. Class/Weight computation: For each link, compute the pressure for each class

            $W_{i,j}^*(t), k_{i,j}^*(t) = max/argmax_k[ Q_{i,k}(t) - Q_{j,k}(t) + b_{i,j,k}]$
            where b_{i,j,k} is the bias for class k packets from node i to node j

        2. Scheduling: Schedule a single class for each node

            For each node n:
                a. Get the links that start at node n
                b. Schedule the link corresponding to:
                    $max_{j in N(i)} {W_{i,j}^*(t) * C_{i,j}(t) - V P_{i,j}(t)}$
                    where V is the power penalty, P_{i,j}(t) is the power of the link from i to j, and C_{i,j}(t) is the
                     capacity of the link from i to j
                c. Transmit the maximum number of class k_{i,j}^*(t) packets that can be transmitted over the chosen link

        Resulting action will be an M,K matrix where action[m,k] denotes the number of packets attempted to transmit
        from the start node of link m to the end node of link m for class k
            - Should only have one non-zero element per row
            - That element should be equal to the capacity of the link
            - The amount of packets actual delivered is min(Q[start_nodes], action[m,k])

        """
        # Get the current action mask to prevent routing to nodes that are not allowed for each class
        mask = self._get_mask()

        # Initialize action tensor
        action = torch.zeros([self.M, self.K])

        # Compute the pressure over each link for each class
        # Mask ensures that the pressure is 0 for classes that are not allowed to be sent
        # pressure = (Q[self.start_nodes] - Q[self.end_nodes]  + self.bias) * mask[:, 1:]
        # Version below is for drift-plus-penalty approach for power-delay tradeoff
        pressure = (self.Q[self.start_nodes] - self.Q[self.end_nodes] + self.bias) * mask[:, 1:]

        # Find the class with the maximum pressure for each link
        weights, chosen_class = torch.max(pressure, dim=1, keepdim=True)

        # For each node, schedule a single class corresponding to the maxweight
        for n in range(self.N):
            # get the links that start at node n
            links = self.outgoing_links.get(n, [])
            if len(links) == 0:
                continue
            if self.power_penalty == 0:
                # get the classes that have the highest pressure
                argmax_weight = torch.argmax(weights[links]*self.cap[links].unsqueeze(-1), dim=0)
            else:
                argmax_weight = torch.argmax(
                    weights[links]*self.cap[links].unsqueeze(-1) - self.power_penalty * self.link_power[links].unsqueeze(-1), dim=0)

            if weights[links[argmax_weight]] > 0:
                action[links[argmax_weight], chosen_class[links[argmax_weight]]] = self.cap[links[argmax_weight]]

        # Prevent trying to send packets from empty queues with mask
        action = action * mask[:, 1:]

        # Apply Backpressure
        start_Q = self.Q.sum().item()
        to_transmit = torch.min(action, self.Q[self.start_nodes])
        self.Q.index_add_(0, self.start_nodes, -to_transmit)
        self.Q.index_add_(0, self.end_nodes, to_transmit)
        end_Q = self.Q.sum().item()
        if end_Q - start_Q > 0:
            raise ValueError(
                f"Backpressure with interference increased the total number of packets in the network by {end_Q - start_Q}")

        return action


    def convert_action(self, action):
        """
        Converts a (M,K) action tensor to a representation showing
        start_node: [end_node, action]
        :param action:
        :return:
        """
        if action.shape[1] == self.K + 1:
            action = action[:,1:]
        return {self.link_map[m]: action[m].tolist() for m in range(self.M)}


    def get_net_spec(self, include = ["sp_dist", "bias"]):
        return self.get_rep(include = include)

    def get_rep(self, features = ("link_rates","destination","arrival_rates"))-> TensorDict:
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

        edge_attr = torch.stack(edge_features, dim = 2)
        X = torch.stack(node_features, dim = 2)


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

    def _get_observation(self, action = None, reset = False):
        # Step 3: Deliver all packets that have reached their destination
        departures = self._process_departures()

        # Compute the utilized power = link_power * (sum(action, dim=1)>=1)
        if action is not None:
            active_links = torch.sum(action, dim=1) > 0
            power = torch.sum(self.link_power * active_links)
        else:
            power = torch.tensor(0, dtype = torch.float32)

        # Step 4: Compute reward, with shape (1,)
        reward = -torch.sum(self.Q).reshape([1])
        backlog = torch.sum(self.Q).reshape([1])

        # Step 5: Check if the episode should be truncated
        terminated = False
        if self.max_backlog is not None and -reward > self.max_backlog:
            # reward = -torch.Tensor(100)
            terminated = True

        # Step 6: Generate new arrivals
        arrivals = self._process_arrivals()

        # Step 7: Generate new link capacities
        self._sim_capacities()

        # Step 8: Generate Next Mask
        self.mask = self._get_mask()

        td = TensorDict({
            "Q": self.Q.clone(),
            "backlog": backlog,
            "power": power,
            "cap": self.cap.clone(),
            "departures": departures,
            "arrivals": arrivals,
            "mask": self._get_mask().clone(),
            "sp_dist": self.sp_dist,
            "edge_index": self.edge_index,
            "link_rates": self.link_rates,
            "arrival_rates": self.arrival_rates,
            "context_id": self.context_id,
            "terminated": torch.tensor([terminated], dtype=bool),

        }, batch_size=self.batch_size)
        if action is not None:
            td.set("action", action)


        if not reset:
            td.set("reward", reward)
        return td


    def _get_valid_action(self, action: torch.Tensor):
        """
        WARNING: THIS METHOD TAKES A LONG TIME TO RUN SO CARE SHOULD BE TAKEN TO AVOID HAVING TO CALL IT

        action is an (M,K) tensor where (m,k) denotes the number of packets to transmits over start node of link m to end node of link m of class k
        return a valid action where (m, k) is not greater than the number of packets in the start node of link m
        and where the sum over K of (m,k) is not greater than the capacity of link m

        How do we not send ghost packets? I.e. if the sum of packets that action wants to transmit from Q[i,k] is greater than Q[i,k]
        then we should only send Q[i,k] packets. But we should also not send more packets than the capacity of the link

        We have two constraints.
        1. The sum of packets transmitted over a link cannot exceed the capacity of the link
        2. The sum of packets transmitted from a particular queue cannot exceed the number of packets in the queue

        To implement this:
        1. Get the minimum of the action and the number of packets in the start node -> valid_action
        2. For each Q[n,k] the sum over j in self.outgoing_links[n]of valid_action[j,k] cannot exceed Q[n,k] -> valid_action
        3. For each link m, the sum over k of valid_action[m,k] cannot exceed the capacity of the link -> valid_action

        :param action:
        :return:
        """
        if action.dim() == 1:
            # convert to (M,K+1) where
            # new_action[m,k] = cap[m] if action[m] = k
            new_action = torch.zeros([self.M, self.K+1])
            new_action[torch.arange(self.M), action] = self.cap
            action = new_action
        if action.shape[1] == self.K + 1:
            action = action[:,1:]
        action = self.base_mask* action
        valid_action = torch.zeros_like(action)
        # copy of capacity
        residual_capacity = self.cap.clone()
        # copy of Q
        residual_packets = self.Q.clone()
        # Loop through and randomize priority
        for m in torch.randperm(self.M):
            if residual_capacity[m] == 0: # speedup
                continue
            for k in torch.randperm(self.K): # speedup
                if residual_packets[self.start_nodes[m],k] == 0:
                    continue
                # valid_action[m,k] must be the minimum of:
                # 1. action[m,k] - the number of class k packets at link m to transmit
                # 2. residual_packets[self.start_nodes[m],k] - the number of class k packets at the start node of link m
                # 3. residual_capacity[m] - the remaining capacity of link m
                valid_action[m,k] = torch.min(action[m,k], torch.min(residual_packets[self.start_nodes[m],k], residual_capacity[m]))
                residual_packets[self.start_nodes[m],k] = torch.clamp(residual_packets[self.start_nodes[m],k]-valid_action[m,k], min = 0)
                residual_capacity[m] = torch.clamp(residual_capacity[m] - valid_action[m,k], min = 0)
        return valid_action


    def _get_random_valid_action(self):
        # samples a random action where each action[m,k] < cap[m]
        action = torch.ones([self.M, self.K])*self.cap.max()
        return self._get_valid_action(action)

    """
    Need to create a set of methods for:
    1. Checking if there is a path from node i to the destination node of class k
    2. If not, updating the mask such that all actions that try to send class k packets from node i 
        to a node that is not connected to the destination node of class k are set to zero
    """

    def _init_shortest_path_dist(self):
        """
        Get the shortest_path from each node to the destination node of each class
        if there is not a shortest path  node i to destination k
        then set the shortest_path_length to a very large number

        :return:
        """

        for source, dest_dict in nx.all_pairs_shortest_path_length(self.graphx):
            self.shortest_path_dist[source] = {}
            for n in range(self.N):
                if n in dest_dict:
                    self.shortest_path_dist[source][n] = dest_dict[n]
                else:
                    self.shortest_path_dist[source][n] = self.max_shortest_path
        # Convert to tensor
        for n in range(self.N):
            for k in range(self.K):
                try:
                    self.sp_dist[n,k] = self.shortest_path_dist[n][self.destination_map[k]]
                except KeyError:
                    self.sp_dist[n,k] = self.max_shortest_path
        return self.sp_dist

    def _init_weighted_shortest_path_dist(self):

        for source, dest_dict in nx.all_pairs_dijkstra_path_length(self.graphx):
            self.weighted_shortest_path_dist[source] = {}
            for n in range(self.N):
                if n in dest_dict:
                    self.weighted_shortest_path_dist[source][n] = dest_dict[n]
                else:
                    self.weighted_shortest_path_dist[source][n] = self.max_shortest_path
        # Convert to tensor
        for n in range(self.N):
            for k in range(self.K):
                try:
                    self.weighted_sp_dist[n,k] = self.weighted_shortest_path_dist[n][self.destination_map[k]]
                except KeyError:
                    self.weighted_sp_dist[n,k] = self.max_shortest_path
        return


    def _init_base_mask(self):
        """
        Creates an (M,K) base_mask where base_mask[m,k] is true if the end node
        of link m can reach the destination of class k
        We can check this by seeing if self.shortest_path_dist[self.end_nodes[m]][self.destination_map[k]] < self.max_shortest_path]
        :return:
        """
        for m in range(self.M):
            for k in range(self.K):
                self.base_mask[m,k] = self.shortest_path_dist[self.end_nodes[m].item()][self.destination_map[k]] < self.max_shortest_path

    def _get_mask(self):
        """
        In sets mask (m,k) to False if the start node of link m has no packets of class k
        :return:
        """
        mask = self.Q[self.start_nodes] > 0 # Prevents sending packets from nodes with no packets
        mask = torch.logical_and(mask, self.base_mask) # Prevents sending packets to nodes that cannot reach the destination
        any_true = torch.any(mask, dim = 1,keepdim = True) # Allows idling only if there are no other valid actions
        return torch.concat([~any_true, mask], dim = 1)
        # ones = torch.ones([mask.shape[0],1], dtype = torch.bool)
        # return torch.concat([ones,torch.logical_and(mask, self.base_mask)],dim=1)

    def set_bias(self, bias):
        if bias.dim() > 2:
            bias = bias.squeeze(-1)
        if bias.dim() == 1:
            bias = bias.unsqueeze(-1)
        self.bias = torch.tensor(bias, dtype = torch.float32)
        if bias.shape != (self.M, self.K):
            raise ValueError(f"bias must have shape (M,K) ({self.M}, {self.K}) but input has shape {bias.shape}")

    def get_context_id(self):
        return int(self.context_id.item())






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


