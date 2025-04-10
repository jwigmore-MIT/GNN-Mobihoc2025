from utils import *
import numpy as np
import random
from env_sim.MultiClassMultihopTS import MultiClassMultiHopTS, create_link_weights
from env_sim.MultiClassMultihopBP import MultiClassMultiHopBP, create_sp_bias
import inspect
import concurrent.futures
import time


# set print options for torch
torch.set_printoptions(precision=2, sci_mode=False)

def run_with_timeout(func, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print("Function timed out")
            return None

def generate_random_network(num_nodes,
                            num_classes,
                            topology_types ='barabasi_albert',
                            avg_degree=3,
                            service_rate_sampler=lambda: np.random.randint(1, 10),
                            link_power_sampler=lambda: np.random.randint(1, 5),
                            arrival_rate_sampler=lambda: np.random.uniform(0, 1),
                            arrival_node_density=0.1,
                            seed=None,
                            metrics = True):
    """
    Generate a random network with specified parameters.

    Args:
        num_nodes: Number of nodes in the network
        num_classes: Number of traffic classes
        topology_type: Type of network topology to generate
        avg_degree: Average node degree (for applicable topology types)
        seed: Random seed for reproducibility

    Returns:
        A dictionary containing network configuration
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if isinstance(arrival_node_density, float):
        arrival_node_density = [arrival_node_density] * num_classes

    while True:
        if isinstance(topology_types, list):
            # randomly select a topology type from the list
            topology_type = random.choice(topology_types)
        else:
            topology_type = topology_types

        # Generate network topology
        if topology_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(num_nodes, int(avg_degree / 2), seed=seed)
        elif topology_type == 'erdos_renyi':
            p = avg_degree / (num_nodes - 1)
            G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)

        elif topology_type == 'watts_strogatz':
            k = int(avg_degree)
            p = 0.1
            G = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
        elif topology_type == 'grid':
            side = int(np.sqrt(num_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")

        # check if graph is connected if undirected or strongly connected if directed
        if not G.is_directed() and nx.is_connected(G):
            break
        elif G.is_directed() and nx.is_strongly_connected(G):
            break

    # Convert to directed graph if not already
    if not G.is_directed():
        G = G.to_directed()

    # Generate link information
    link_info = []
    # save the link rates of link (u,v) to assign to (v,u) as well
    service_rate_dict ={}
    link_power_dict = {}
    for i, (u, v) in enumerate(G.edges()):
        # Assign random service rate to each link (between 1 and 10)
        if (v,u) in service_rate_dict:
            service_rate = service_rate_dict[(v, u)]
            link_power = link_power_dict[(v, u)]
        else:
            service_rate = service_rate_sampler()
            link_power = link_power_sampler()
            service_rate_dict[(u, v)] = service_rate
            link_power_dict[(u, v)] = link_power

        link_info.append({
            "start": u,
            "end": v,
            "rate": float(service_rate),
            "power": float(link_power)
        })



    # Generate class information
    class_info = {}
    for c in range(num_classes):
        # Select a random destination node for this class
        destination = random.randint(0, num_nodes - 1)
        class_info[destination] = {}

        # get the sum of arrival rates going to the destination node
        incoming_service_rate = sum([link["rate"] for link in link_info if link["end"] == destination])
        sum_arrival_rate = 0
        # Get shortest path to destination node from every node

        # Generate arrival rates from other nodes to this destination
        for n in range(num_nodes):
            if n != destination and np.random.uniform() < arrival_node_density[c]:
                # Assign random arrival rate (between 0 and 1)
                arrival_rate = arrival_rate_sampler()
                # if the arrival rate is > the service rate of all outgoing links, resample
                attempts = 0
                while arrival_rate > sum([link["rate"] for link in link_info if link["start"] == n]):
                    if attempts > 30:
                        continue # skip this node
                    arrival_rate = arrival_rate_sampler()
                    attempts +=1
                if sum_arrival_rate + arrival_rate < incoming_service_rate:
                    class_info[destination][n] = round(float(arrival_rate),2)
                sum_arrival_rate += arrival_rate

        if len(class_info[destination]) == 0:
            # If no arrival rates assigned, assign a random one
            n = random.randint(0, num_nodes - 1)
            while n == destination:
                n = random.randint(0, num_nodes - 1)
            arrival_rate = np.random.uniform(0, 1)
            class_info[destination][n] = round(float(arrival_rate),2)

    # Create the network configuration
    network_config = {
        "nodes": num_nodes,
        "link_info": link_info,
        "class_info": class_info,
        "link_distribution": "fixed",
        "arrival_distribution": "poisson",
        "seed": seed if seed is not None else random.randint(0, 10000),
        "topology_type": topology_type,
    }

    network_config["metrics"] = get_metrics(network_config, G)



    return network_config, G

def get_metrics(network_config, G):
    # Basic network metrics
    metrics = {
        "num_nodes": len(G.nodes()),
        "num_links": len(G.edges()),
        "num_classes": len(network_config["class_info"])
    }

    # Graph Topology Metrics
    metrics.update({
        "topology_type": network_config["topology_type"],
        "avg_in_degree": float(sum(dict(G.in_degree()).values())) / len(G.nodes()),
        "avg_out_degree": float(sum(dict(G.out_degree()).values())) / len(G.nodes()),
        "density": nx.density(G),
        "diameter": nx.diameter(G) if nx.is_strongly_connected(G) else float('inf'),
    })

    # Class metrics
    class_arrival_rates = {}
    for dest, sources in network_config["class_info"].items():
        class_arrival_rates[dest] = sum(sources.values())

    metrics.update({
        "total_arrival_rate": sum(class_arrival_rates.values()),
        "max_class_arrival_rate": max(class_arrival_rates.values()),
        "min_class_arrival_rate": min(class_arrival_rates.values()),
        "std_class_arrival_rate": np.round(np.std(list(class_arrival_rates.values())),2)
    })

    link_rates = [link["rate"] for link in network_config["link_info"]]
    metrics.update({
        "avg_link_rate": np.round(np.mean(link_rates),2),
        "min_link_rate": min(link_rates),
        "max_link_rate": max(link_rates),
        "std_link_rate": np.round(np.std(link_rates),2)
    })

    # For each source-destination pair, count alternative paths
    path_diversities = []
    for dest, sources in network_config["class_info"].items():
        for source in sources:
            # Count number of edge-disjoint paths
            try:
                diversity = len(list(nx.edge_disjoint_paths(G, int(source), int(dest))))
                path_diversities.append(diversity)
            except nx.NetworkXNoPath:
                path_diversities.append(0)

    metrics.update({
        "avg_path_diversity": np.mean(path_diversities) if path_diversities else 0,
        "min_path_diversity": min(path_diversities) if path_diversities else 0
    })

    # Calculate maximum flow between each source-destination pair
    flow_ratios = []
    # set capacity for each edge in G equal to the link rate
    for u, v in G.edges():
        G[u][v]["capacity"] = next(link["rate"] for link in network_config["link_info"] if link["start"] == u and link["end"] == v)
    for dest, sources in network_config["class_info"].items():
        dest = int(dest)
        for source, rate in sources.items():
            source = int(source)
            try:
                # Create a flow network with capacity = link rate
                flow_value = nx.maximum_flow_value(G, source, dest,
                                                   capacity="capacity")
                flow_ratios.append(rate / flow_value if flow_value > 0 else float('inf'))
            except:
                flow_ratios.append(float('inf'))

    metrics.update({
        "avg_flow_ratio": np.round(np.mean([r for r in flow_ratios if r != float('inf')]),2) if flow_ratios else 0,
        "max_flow_ratio": max([r for r in flow_ratios if r != float('inf')]) if flow_ratios else 0,
        "num_unsatisfiable_flows": sum(1 for r in flow_ratios if r > 1 or r == float('inf'))
    })

    # format all metrics to be floats

    return metrics


def print_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\n")


def test_network(network_config, policies, num_evals, max_steps, min_throughput, max_throughput, min_min_backlog):
    results = {}
    method_dict = {"CWTS": "capacity_weighted_shortest_path",
                   "SPTS": "shortest_path",
                   "SPBP2": "capacity_weighted_shortest_path", }

    for poli in policies:
        results[poli] = []
        for n in range(num_evals):
            if poli in ["BP", "SPBP", "SPBP2"]:
                network_config["action_func"] = None
                env = MultiClassMultiHopBP(**network_config)
                if poli == "SPBP":
                    bias, _ = create_sp_bias(env, weighted=True)
                    env.set_bias(bias)
                elif poli == "SPBP2":
                    weights = create_link_weights(env, method=method_dict["SPBP2"])
                    env.set_bias(weights)
            elif poli in ["CWTS", "SPTS"]:
                env = MultiClassMultiHopTS(**network_config)
                weights = create_link_weights(env, method=method_dict[poli])
                env.W = weights
            elif "DPP" in poli:  # DPP with power penalty
                power_penalty = float(poli.split("_")[1])
                network_config["action_func"] = "bpi"
                network_config["power_penalty"] = power_penalty
                env = MultiClassMultiHopBP(**network_config)
            td = env.rollout(max_steps)
            results[poli].append(TensorDict({
                "mean_backlog": td["backlog"].mean(),
                "departures": td["next", "departures"].sum(),
                "arrivals": td["arrivals"].sum(),
                "throughput": td["next", "departures"].sum() / td["arrivals"].sum(),
                "mean_power": td["next"].get("power", torch.tensor([0.0])).float().mean()
            }))
            # Skip current context if the throughput or backlog is outside the specified range
            if poli == "SPBP":
                if results["SPBP"][-1]["throughput"] < min_throughput:
                    print(f"Skipping current context because SPBP throughput is less than {min_throughput}")
                    print_metrics(network_config["metrics"])
                    return {}
                elif results["SPBP"][-1]["throughput"] > max_throughput:
                    print(f"Skipping current context because SPBP throughput is greater than {max_throughput}")
                    print_metrics(network_config["metrics"])
                    return {}
                elif results["SPBP"][-1]["mean_backlog"] < min_min_backlog:
                    print(f"Skipping current context because min backlog is less than {min_min_backlog}")
                    print_metrics(network_config["metrics"])
                    return {}
            elif "DPP" in poli:
                if results[poli][-1]["throughput"] < min_throughput:
                    print(f"Skipping current context because {poli} throughput is less than {min_throughput}")
                    print_metrics(network_config["metrics"])
                    return {}
                elif results[poli][-1]["throughput"] > max_throughput:
                    print(f"Skipping current context because {poli} throughput is greater than {max_throughput}")
                    print_metrics(network_config["metrics"])
                    return {}
                elif results[poli][-1]["mean_backlog"] < min_min_backlog:
                    print(f"Skipping current context because min backlog is less than {min_min_backlog}")
                    print_metrics(network_config["metrics"])
                    return {}
    return results


def create_context_set(
                       num_contexts,
                       name = None,
                       topology_types = 'barabasi_albert',
                       avg_degree_sampler = lambda: np.random.randint(3,5),
                       num_node_sampler = lambda: np.random.randint(10, 50),
                       destination_density_sampler = lambda: np.random.uniform(0.1, 0.3),
                       arrival_node_density_sampler = lambda: np.random.uniform(0.1, 0.3),
                       service_rate_sampler=lambda: np.random.randint(1, 10),
                       link_power_sampler=lambda: np.random.uniform(1, 5),
                       arrival_rate_sampler=lambda: np.random.uniform(0, 1),
                       policies = ["capacity_weighted_shortest_path", "shortest_path", "BP", "SPBP"],
                       power_penalty = 1.0,
                       num_evals = 3,
                       max_steps = 1000,
                       min_throughput = 0.9,
                       max_throughput = 1.0,
                       min_min_backlog = 30,
                       save = True,
                       plot_graph = False,
                       metrics = True,
                       topology_timeout_seconds = 120
                       ):
    """
    Samples num_contexts random network configurations.

    :param num_contexts:
    :param topology_type:
    :param avg_degree:
    :param num_node_sampler:
    :param destination_density:
    :param arrival_node_density:
    :param service_rate_sampler:
    :param arrival_rate_sampler:
    :return:
    """

    if save:
        # Create a folder for the context set
        datetime_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        if name is None:
            folder_name = f"{topology_types}_{datetime_str}"
        else:
            folder_name = name
        os.makedirs(os.path.join(CONTEXT_SETS_DIR, folder_name), exist_ok=True)
        # print directory
        print(f"Created directory {os.path.join(CONTEXT_SETS_DIR, folder_name)}")

        # Create a file to store the context set sampling parameters
        context_set_file = os.path.join(CONTEXT_SETS_DIR, folder_name, f"context_set_{topology_types}_{datetime_str}_sampling_params.json")
        with open(context_set_file, 'w') as f:
            json.dump({
                "topology_type": topology_types,
                "avg_degree_sampler": inspect.getsource(avg_degree_sampler).strip(),
                "num_node_sampler": inspect.getsource(num_node_sampler).strip(),
                "destination_density_sampler": inspect.getsource(destination_density_sampler).strip(),
                "arrival_node_density_sampler": inspect.getsource(arrival_node_density_sampler).strip(),
                "service_rate_sampler": inspect.getsource(service_rate_sampler).strip(),
                "link_power_sampler": inspect.getsource(link_power_sampler).strip(),
                "arrival_rate_sampler": inspect.getsource(arrival_rate_sampler).strip(),
                "num_evals": num_evals,
                "max_steps": max_steps
            }, f)

    network_configs = {}

    while len(network_configs) < num_contexts:
        i = len(network_configs)
        num_nodes = num_node_sampler()
        destination_density = destination_density_sampler()
        num_classes = max(int(num_nodes * destination_density),2)
        arrival_node_density = [arrival_node_density_sampler()]*num_classes
        avg_degree = avg_degree_sampler()
        # create a timeout for the network generation

        topology_results = run_with_timeout(lambda:
            generate_random_network(num_nodes=num_nodes,
                                                    num_classes=num_classes,
                                                    topology_types=topology_types,
                                                    avg_degree=avg_degree,
                                                    service_rate_sampler=service_rate_sampler,
                                                    link_power_sampler=link_power_sampler,
                                                    arrival_rate_sampler=arrival_rate_sampler,
                                                    arrival_node_density=arrival_node_density,
                                                    seed=random.randint(0, 10000),
                                                    metrics = metrics)
            , topology_timeout_seconds)
        if topology_results is None:
            print(f"Topology generation timed out ({topology_timeout_seconds}s) for context {i+1}")
            continue
        network_config, G = topology_results
        if plot_graph:
            pos = nx.spring_layout(G, iterations=100, seed=1, k=1, scale=2)
            nx.draw(G, pos, with_labels=True, font_weight='bold')
            plt.show()
        # test the network eval_num times
        results = test_network(network_config, policies, num_evals, max_steps, min_throughput, max_throughput, min_min_backlog)
        if len(results) == 0:
            continue



        avg_results = {}
        for poli in results:
            avg_results[poli] = TensorDict.stack(results[poli]).mean()

        network_config["baselines"] = {}
        network_config["baselines"]["max_steps"] = max_steps
        print(f"Context {i+1} of {num_contexts}")
        if metrics:
            # print all metrics for the network
            for key, value in network_config["metrics"].items():
                print(f"{key}: {value}")
        print("Throughput and Backlog for each policy")
        for poli in policies:
            network_config["baselines"][poli] = {
                "backlog": round(avg_results[poli]["mean_backlog"].item(), 2),
                "throughput": round(avg_results[poli]["throughput"].item(), 3),
                "power": round(avg_results[poli]["mean_power"].item(), 2)
            }
            network_config["metrics"][f"{poli}_backlog"] = round(avg_results[poli]["mean_backlog"].item(), 2)
            network_config["metrics"][f"{poli}_throughput"] = round(avg_results[poli]["throughput"].item(), 3)
            network_config["metrics"][f"{poli}_power"] = round(avg_results[poli]["mean_power"].item(), 2)
            # print the throughput, backlog, and power
            print(f"{poli} : Throughput = {round(avg_results[poli]['throughput'].item(), 2)}"
                  f"; Backlog = {round(avg_results[poli]['mean_backlog'].item(), 2)}"
                  f"; Power = {round(avg_results[poli]['mean_power'].item(), 2)}"
                  )


        # Get the maximum throughput of any policy
        if "SPBP" in policies:
            bp_throughput = avg_results["SPBP"]["throughput"].item()
            if bp_throughput < min_throughput:
                print(f"Skipping current context {i+1} because SPBP throughput is less than {min_throughput}")
                print("\n")
                continue
            elif bp_throughput > max_throughput:
                print(f"Skipping current context {i+1} because SPBP throughput is greater than {max_throughput}")
                print("\n")
                continue
            min_backlog = np.min([avg_results[poli]["mean_backlog"].item() for poli in policies])
            if min_backlog < min_min_backlog:
                print(f"Skipping current context {i+1} because min backlog is less than {min_min_backlog}")
                print("\n")
                continue
            if network_config["metrics"]["num_unsatisfiable_flows"] > 0:
                print(f"Skipping current context {i+1} because there are unsatisfiable flows")
                print("\n")
                continue


        # # Add network statistics to the network configuration
        # network_config["network_statistics"] = {
        #     "nodes": num_nodes,
        #     "edges": len(G.edges()),
        #     "classes": num_classes,
        #     "diameter": nx.diameter(G),
        #     "avg_degree": sum(dict(G.degree()).values()) / num_nodes,
        # }


        if save:
            # Save the network configuration
            output_file = os.path.join(CONTEXT_SETS_DIR, folder_name, f"network_{network_config["topology_type"]}_{datetime_str}_Env{i}.json")
            with open(output_file, 'w') as file:
                json.dump(network_config, file)
        network_configs[i] = network_config

    context_set_stats = compute_metric_stats([network_config["metrics"] for network_config in network_configs.values()])
    if save:
        with open(os.path.join(CONTEXT_SETS_DIR, folder_name, f"context_set_{topology_types}_{datetime_str}_metrics.json"), 'w') as f:
            json.dump(context_set_stats, f)
    # print context set statistics
    print("Context Set Statistics")
    for key, value in context_set_stats.items():
        print(f"{key}: {value}")

    return network_configs
if __name__ == "__main__":
    network_config, G = generate_random_network(num_nodes=30,
                                                num_classes=5,
                                                topology_type = 'watts_strogatz',
                                                arrival_node_density=0.2,
                                                avg_degree=4,
                                                seed = random.randint(0, 10000))
    env = MultiClassMultiHopTS(**network_config)
    for k, dest in enumerate(network_config["class_info"].keys()):
        source_nodes = [source for source in network_config["class_info"][dest].keys()]
        destination_nodes = dest
        # set edge_attr to be the link rates
        edge_attr = env.link_rates
        node_attr = env.arrival_rates[:,dest]
        fig, ax= plot_single_nx_graph(G,  edge_attr = edge_attr, node_attr = node_attr, source_nodes = source_nodes, dest_nodes = destination_nodes, title = f"Class {k} Traffic")
        fig.show()


    network_config["baselines"] = {}
    max_steps = 1000
    results = {}
    policies = ["capacity_weighted_shortest_path", "shortest_path"]
    tds = {}
    for pol in policies:
        env = MultiClassMultiHopTS(**network_config)
        weights = create_link_weights(env, method = pol)
        env.W = weights
        td = env.rollout(max_steps)
        results[pol]=TensorDict({
            "mean_backlog": td["backlog"].mean(),
            "departures": td["next", "departures"].sum(),
            "arrivals": td["arrivals"].sum(),
            "throughput": td["next", "departures"].sum() / td["arrivals"].sum()
        })
        tds[pol] = td
        network_config["baselines"][pol] = {
            "backlog": round(td["backlog"].mean().item(), 2),
            "throughput": round(td["next", "departures"].sum().item() / td["arrivals"].sum().item(), 2)
        }

    network_config["baselines"]["max_steps"] = max_steps



    # plot the time average backlog
    fig, ax = plt.subplots()
    for pol in policies:
        ax.plot(tds[pol]["backlog"], label = f"{pol} : {round(results[pol]['throughput'].item(),2)}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Backlog")
    ax.legend()
    fig.show()




#
# # parameters
# N = 20
# num_classes = 2
# p=0.1
#
# datetime_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
# folder_name = f"gnp_N{N}_K{num_classes}_{datetime_str}"
# os.makedirs(os.path.join(ENVS_DIR, folder_name), exist_ok = True)
# output_file = os.path.join(ENVS_DIR, folder_name, f"max_capacity_gnp_N{N}_K{num_classes}_{datetime_str}.json")
# network_config = create_gnp_network(N, p, num_classes, digraph = True, link_rate_sampler=lambda: random.choice([1,2,3,4]))
# network_config["link_distribution"] = "fixed"
# network_config["arrival_distribution"] = "poisson"
#
# G = nx.DiGraph()
# G.add_nodes_from(network_config["nodes"])
# G.add_edges_from([(link["start"], link["end"]) for link in network_config["link_info"]])
# source_nodes = {}
# destination_nodes = {}
# env = MultiClassMultiHopBPGen(**network_config)
# for k, dest in enumerate(network_config["class_info"].keys()):
#     source_nodes[k] = [source for source in network_config["class_info"][dest].keys()]
#     destination_nodes[k] = dest
#     # set edge_attr to be the link rates
#     edge_attr = env.link_rates
#     node_attr = env.arrival_rates[:,dest]
#     fig, ax= plot_single_nx_graph(G,  edge_attr = edge_attr, node_attr = node_attr, source_nodes = source_nodes[k], dest_nodes = destination_nodes[k], title = f"Class {k} Traffic")
#     fig.show()
#
# # First test the network at the base arrival rate
# # td, sp_td = test_network(network_config, title = "Base Arrival Rate")
#
# max_steps = 1000
# results = defaultdict(list)
# policies = ["SP_10"]
# for pol in policies:
#     results[pol] = []
# p_scales = np.arange(0.2, 4.0, 0.2)
# scales = []
# for scale in list(p_scales):
#     print("Testing Network at Scale = ", scale)
#     scales.append(scale)
#     new_network_config = scale_arrival_rates(network_config, scale = scale)
#     test_results = test_network(new_network_config, policies= policies, title = f"Scale = {round(scale,1)}", max_steps=max_steps)
#     for pol in results:
#         results[pol].append(test_results[pol])
#         if test_results[pol]["delivery_rate"] < 0.9:
#             break
#
#
#
# for pol in results:
#     results[pol] = TensorDict.stack(results[pol])
#
#
# # plot throughput as a function of scale
# fig, ax = plt.subplots()
# for pol in policies:
#     ax.plot(scales, results[pol]["departures"], label = "pol")
# ax.set_xlabel("Scale")
# ax.set_ylabel("Throughput")
# ax.legend()
# fig.suptitle("Throughput vs Arrival Rate Scale")
# fig.show()
#
# # plot departures/arrivals
#
# fig, ax = plt.subplots()
# for pol in policies:
#     ax.plot(scales, results[pol]["delivery_rate"], label = "BP")
# ax.set_xlabel("Scale")
# ax.set_ylabel("Fraction")
# ax.legend()
# fig.suptitle("Throughput vs Arrival Rate Scale")
# fig.show()
#
# # Max Rate will be the first scale where throughput is less than 1-eps
# eps = 0.1
# max_rate = scales[np.where(results[policies[-1]]["delivery_rate"] < 1-eps)[0][0]]
#
# max_capacity_network_config = scale_arrival_rates(network_config, scale = max_rate)
# for pol in policies:
#     low_pol = pol.lower()
#     max_capacity_network_config[low_pol + "_lta"] = round(results[pol][np.where(scales == max_rate)[0][0]]["mean_backlog"].item(), 2)
# max_capacity_network_config["lta_steps"] = max_steps
# #
# # # Save the max capacity network configuration
# with open(output_file, 'w') as file:
#     json.dump(max_capacity_network_config, file)
# #
# #
# #
