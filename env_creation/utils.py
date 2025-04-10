from env_sim.MultiClassMultihopBP import MultiClassMultiHopBP, create_sp_bias
from copy import deepcopy
import datetime
import os
from tensordict import TensorDict
import platform
import random
import json
from collections import defaultdict
from matplotlib import pyplot as plt
from torchrl.envs import ExplorationType, set_exploration_type
import math
import networkx as nx
import matplotlib as mpl
import torch_geometric as pyg
import numpy as np
import  warnings
from torchrl.collectors import MultiSyncDataCollector
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)

opsys = platform.system()
# get the number of cpus
if opsys == "Windows":
    PROJECT_DIR = "C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\"
else:
    PROJECT_DIR = "/home/jwigmore/PycharmProjects/GDRL4Nets"

REL_ENVS_DIR = r"experiments/GNNBiasedBackpressureDevelopment/envs"
REL_CONTEXT_SETS_DIR = r"experiments/GNNBiasedBackpressureDevelopment/context_sets"
ENVS_DIR = os.path.join(PROJECT_DIR, REL_ENVS_DIR)
CONTEXT_SETS_DIR = os.path.join(PROJECT_DIR, REL_CONTEXT_SETS_DIR)



def create_grid_topology(grid_size, link_rate_sampler = lambda: 1, network_config = {}):
    nodes = list(range(grid_size ** 2))
    link_info = []
    # Create links for the grid
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if j < grid_size - 1:  # Horizontal link
                link_info.append({"start": node_id, "end": node_id + 1, "rate": link_rate_sampler()})
                # reverse link
                link_info.append({"start": node_id + 1, "end": node_id, "rate": link_rate_sampler()})
            if i < grid_size - 1:  # Vertical link
                link_info.append({"start": node_id, "end": node_id + grid_size, "rate": link_rate_sampler()})
                # reverse link
                link_info.append({"start": node_id + grid_size, "end": node_id, "rate": link_rate_sampler()})
    network_config["nodes"] = nodes
    network_config["link_info"] = link_info
    return network_config


def create_gnp_network(N, p, K,  digraph = False, link_rate_sampler = lambda: 1, arrival_rate_sampler = lambda: random.random(),
    network_config = {}):

    # sample a random graph topology
    G = nx.fast_gnp_random_graph(N, p, directed=True)
    # Convert to a directed graph if True
    if digraph:
        G = nx.DiGraph(G)
    # plot the graph
    # pos = nx.spring_layout(G, iterations=100, seed=1, k=1, scale=2)
    # nx.draw(G, pos, with_labels=True, font_weight='bold')
    # plt.show()
    # initialize the network configuration information
    nodes = list(G.nodes)
    links = list(G.edges)

    # Sample link rates
    link_info = []
    for link in links:
        link_info.append({"start": link[0], "end": link[1], "rate": link_rate_sampler()})

    # sample K destinations
    valid_destinations = random.sample(list(G.nodes), K)
    class_info = defaultdict(dict)
    # Sample source nodes for each destination
    for destination in valid_destinations:
        source_probability = 1
        # Get all possible source nodes - has a path to the destination node
        sources = list(nx.single_target_shortest_path(G, destination).keys())
        # delete the destination node from the sources
        sources.remove(destination)
        # ensures a quarter of the sources are selected, or at least one source is selected
        if len(sources) < 1:
            continue
        shuffled = random.sample(sources, max(1,len(sources)//4))
        for source in shuffled:
            # sample arrival rates for each source to destination pair
            class_info[destination][source] = round(arrival_rate_sampler(), 2)


    network_config["nodes"] = nodes
    network_config["link_info"] = link_info
    network_config["class_info"] = class_info
    return network_config

def create_random_class_information(network_config, arrival_rate_sampler = random.random, num_classes = 4):
    """
    Randomly sample destination pairs, assign arrival rates, split arrival rates over random source nodes
    :param network_config:
    :param arrival_rate_sampler:
    :return:
    """
    nodes = network_config["nodes"]
    K = num_classes
    valid_destinations = random.sample(nodes, K)
    class_info= defaultdict(dict)
    G = nx.DiGraph()
    for link in network_config["link_info"]:
        G.add_edge(link["start"], link["end"])
    for destination in valid_destinations:
        source_probability = 1
        # Get all possible source nodes - has a path to the destination node
        sources = nx.single_target_shortest_path(G, destination)
        shuffled = random.sample(sources, len(sources))

        for source in shuffled:
            if source != destination and random.random() < source_probability:
                class_info[destination][source] = round(arrival_rate_sampler(), 2)
                source_probability = source_probability / 2
                if source_probability < 0.05:
                    break
    network_config["class_info"] = class_info

    return network_config

def scale_arrival_rates(network_config, scale = 0.5):
    to_del = []
    network_config = deepcopy(network_config)
    for destination in network_config["class_info"]:
        for source in network_config["class_info"][destination]:
            network_config["class_info"][destination][source] = round(scale * network_config["class_info"][destination][source], 2)
            if network_config["class_info"][destination][source] < 0.1:
                to_del.append((destination, source))
    # for destination, source in to_del:
    #     del network_config["class_info"][destination][source]
    return network_config
def create_grid_network_corner_arrivals(grid_size, link_rate, arrival_rate, output_file):
    nodes = list(range(grid_size ** 2))
    link_info = []
    class_info = []

    # Create links for the grid
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if j < grid_size - 1:  # Horizontal link
                link_info.append({"start": node_id, "end": node_id + 1, "rate": link_rate})
                # reverse link
                link_info.append({"start": node_id + 1, "end": node_id, "rate": link_rate})
            if i < grid_size - 1:  # Vertical link
                link_info.append({"start": node_id, "end": node_id + grid_size, "rate": link_rate})
                # reverse link
                link_info.append({"start": node_id + grid_size, "end": node_id, "rate": link_rate})
            if i == 0 and j == 0:
                class_info.append({"source": node_id, "destination": grid_size ** 2 - 1, "rate": arrival_rate})
            elif i == grid_size - 1 and j == grid_size - 1:
                class_info.append({"source": node_id, "destination": 0, "rate": arrival_rate})
            elif i == 0 and j == grid_size - 1:
                class_info.append({"source": node_id, "destination": grid_size * (grid_size - 1), "rate": arrival_rate})
            elif i == grid_size - 1 and j == 0:
                class_info.append({"source": node_id, "destination": grid_size - 1 , "rate": arrival_rate})

    # Create traffic classes, where each source is a corner node and the destination is the opposite corner

    # Create the network configuration dictionary
    network_config = {
        "nodes": nodes,
        "link_distribution": "fixed",
        "arrival_distribution": "poisson",
        "link_info": link_info,
        "class_info": class_info
    }

    # Write the configuration to a YAML file
    with open(output_file, 'w') as file:
        json.dump(network_config, file)
    return network_config

# Example usage




def test_network(network_config, policies = ["BP", "SP"], plot = True,
                 title = None, max_steps = 2000, max_backlog = 3000, return_td = False):
    results = {}
    td_results = {}
    for pol in policies:
        env = MultiClassMultiHopBP(**network_config, max_backlog=max_backlog)

        if "SP" in pol:
            alpha = float(pol.split("_")[-1])
            sp_bias, _ = create_sp_bias(env)
            env.set_bias(alpha*sp_bias)

        td = env.rollout(max_steps = max_steps)
        mean_backlog = td["next", "backlog"].mean().item()
        results[pol] = TensorDict({
                        "mean_backlog": mean_backlog,
                        "departures": td["next", "departures"].sum(),
                        "arrivals": td["arrivals"].sum(),
                        "delivery_rate": td["next", "departures"].sum()/ td["arrivals"].sum()})
        td_results[pol] = td

    if plot:
        fig, ax = plt.subplots()
        for pol in policies:
            ax.plot(compute_lta(td_results[pol]["next", "backlog"]), label = pol)

        ax.set_xlabel("Time")
        ax.set_ylabel("Queue Length")
        ax.legend()
        if title:
            fig.suptitle(title)
        plt.show()
    if return_td:
        return results, td_results
    else:
        return results

def compute_metric_stats(metrics):
    """
    Takes in a list of metric (dicts) and computes the mean, standard deviation, min, and max for each metric
    :param metrics:
    :return:
    """

    metric_stats = {}
    for metric in metrics[0].keys():
        if metric == "topology_type":
            # count each class of topology types
            topology_types = [m[metric] for m in metrics]
            topology_type_counts = {topology_type: topology_types.count(topology_type) for topology_type in topology_types}
            metric_stats[metric] = topology_type_counts
        else:
            metric_stats[metric] = {}
            values = [m[metric] for m in metrics]
            metric_stats[metric]["mean"] = float(np.round(np.mean(values), 3))
            metric_stats[metric]["std"] = float(np.round(np.std(values), 3))
            metric_stats[metric]["min"] = float(np.round(np.min(values), 3))
            metric_stats[metric]["max"] = float(np.round(np.max(values), 3))

    return metric_stats



def get_baselines(env_info, num_rollouts, steps_per_env, seed = 0, alphas = [0]):
    def create_env(env_info, alpha):
        seed = random.randint(0, 100000)
        env = MultiClassMultiHopBP(**env_info, seed = seed)
        sp_bias, _ = create_sp_bias(env)
        env.set_bias(sp_bias * alpha)
        return env

    total_frames = steps_per_env*num_rollouts
    sp_rollouts = {alpha: {} for alpha in alphas}

    for alpha in alphas:
        create_env_funcs = []
        for n in range(num_rollouts):
            create_env_funcs.append(create_env(env_info, alpha))

        print("Starting collection for alpha = ", alpha)
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            collector = MultiSyncDataCollector(
                    create_env_fn = create_env_funcs,
                    frames_per_batch = total_frames,
                    reset_at_each_iter = False,
            )

            alpha_rollout = collector.next()
            # compute the lta backlog for each rollout within alpha_rollout
            ltas = torch.stack([compute_lta(alpha_rollout["backlog"][i]) for i in range(num_rollouts)])
            sp_rollouts[alpha] = {
            "ltas": ltas,
            "mean_lta": ltas.mean(dim = 0),
            "std_lta": ltas.std(dim = 0)
            }

        collector.shutdown()


    #plot the performance for each alpha
    fig, ax = plt.subplots()
    for alpha in alphas:
        ax.plot(sp_rollouts[alpha]["mean_lta"], label=f"SP Biased {alpha}")
        ax.fill_between(range(len(sp_rollouts[alpha]["mean_lta"])),
                        sp_rollouts[alpha]["mean_lta"] - sp_rollouts[alpha]["std_lta"],
                        sp_rollouts[alpha]["mean_lta"] + sp_rollouts[alpha]["std_lta"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Length")
    ax.legend()
    plt.show()
    #
    for alpha in alphas:
        env_info[f"sp{alpha}_lta"] = round(sp_rollouts[alpha]["mean_lta"][-1].item(), 2)
        env_info[f"sp{alpha}_lta_std"] = round(sp_rollouts[alpha]["std_lta"][-1].item(), 2)

    env_info["lta_steps"] = total_frames//num_rollouts
    return env_info, sp_rollouts



def plot_nx_graph(graph, edge_attr=None, node_attr=None, K=None, title="", subtitle="", erange=None, vrange=None, source_nodes=[], dest_nodes=[], transform=None):

    if edge_attr is not None and edge_attr.dim() > 2:
        edge_attr = edge_attr.squeeze(-1)
    if K is None:
        K = edge_attr.shape[1]

    rows = int(math.ceil(math.sqrt(K)))
    cols = int(math.ceil(K / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    cmp = plt.cm.bwr_r
    vcmp = plt.cm.winter_r
    if erange is not None:
        emin_val, emax_val = erange
    else:
        emin_val, emax_val = -10, 10
    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin, vmax = None, None
    enorm = mpl.colors.Normalize(vmin=emin_val, vmax=emax_val)
    vnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    all_axes = axes.flatten() if K > 1 else [axes]
    for k, ax in enumerate(all_axes):
        if k >= K:
            break
        if edge_attr is not None:
            graph.edge_attr = edge_attr[:, k].unsqueeze(-1)
        if node_attr is not None:
            graph.node_attr = node_attr[:, k].unsqueeze(-1)
        nx_graph = pyg.utils.to_networkx(graph, edge_attrs=["edge_attr"], node_attrs=["node_attr"], to_undirected=False)
        pos = nx.kamada_kawai_layout(nx_graph)

        if node_attr is not None:
            node_color = [node[1]["node_attr"][0] for node in nx_graph.nodes(data=True)]
            if transform is not None:
                node_color = transform(np.array(node_color))
            if vmin is None:
                vmin = min(node_color)
            if vmax is None:
                med_color = sorted(node_color)[len(node_color) // 2]
                vmax = 2 * med_color

            other_nodes = [node for node in nx_graph.nodes() if node not in source_nodes and node not in dest_nodes]
            other_node_colors = [node[1]["node_attr"][0] for node in nx_graph.nodes(data=True) if node[0] not in source_nodes and node[0] not in dest_nodes]
            if transform is not None:
                other_node_colors = transform(np.array(other_node_colors))
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=other_nodes, node_size=600, alpha=0.5, node_color=other_node_colors, cmap=vcmp, vmin=vmin, vmax=vmax)

            source_node_colors = [node[1]["node_attr"][0] for node in nx_graph.nodes(data=True) if node[0] in source_nodes]
            if transform is not None:
                source_node_colors = transform(np.array(source_node_colors))
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=source_nodes, node_size=2000, node_shape='^', node_color=source_node_colors, cmap=vcmp, vmin=vmin, vmax=vmax)

            dest_node_colors = [node[1]["node_attr"][0] for node in nx_graph.nodes(data=True) if node[0] in dest_nodes]
            if transform is not None:
                dest_node_colors = transform(np.array(dest_node_colors))
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=dest_nodes, node_size=3000, node_shape='*', node_color=dest_node_colors, cmap=vcmp, vmin=vmin, vmax=vmax)

            nx.draw_networkx_labels(nx_graph, pos, ax=ax, labels={i: f"{node_attr[i, k].item():.{1}f}" for i in range(node_attr.shape[0])})
        else:
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=100)
            nx.draw_networkx_labels(nx_graph, pos, ax=ax)
        if edge_attr is not None:
            edge_color = [edge[2]["edge_attr"][0] for edge in nx_graph.edges(data=True)]
            edges = nx.draw_networkx_edges(nx_graph, pos, ax=ax, connectionstyle="arc3,rad=0.2", edge_color=edge_color, edge_cmap=cmp, edge_vmin=emin_val, edge_vmax=emax_val)
            labels = {edge: round(attr[0], 2) for edge, attr in nx.get_edge_attributes(nx_graph, "edge_attr").items()}
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, ax=ax, connectionstyle="arc3,rad=0.2")
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax=ax, connectionstyle="arc3,rad=0.2")
            edges = None
        ax.set_title(f"Class {k}")
    if edges is not None:
        pc = mpl.collections.PatchCollection(edges, cmap=cmp, norm=enorm)
        pc.set_array(edge_color)
    else:
        pc = None
    fig.suptitle(f"{title}\n{subtitle}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if pc is not None:
        fig.colorbar(pc, ax=axes, orientation='vertical', label="Edge Weight")

    return fig, axes



def plot_single_nx_graph(graph, edge_attr=None, node_attr=None, title="", subtitle="",
                         erange=None, vrange=None, source_nodes=[], dest_nodes=[], transform=None,
                         vertex_cmp = None, edge_cmp = None, show_colorbar = True):

    # Make sure dim of node_attr ==1 and edge_attr == 1
    if edge_attr is not None and edge_attr.dim() > 2:
        edge_attr = edge_attr.squeeze(-1)
    if node_attr is not None and node_attr.dim() > 2:
        node_attr = node_attr.squeeze(-1)


    fig, ax = plt.subplots(figsize=(20, 20))
    if edge_cmp is None:
        edge_cmp = plt.cm.bwr_r
    elif isinstance(edge_cmp, str):
        edge_cmp = getattr(plt.cm, edge_cmp)
    if vertex_cmp is None:
        vertex_cmp = plt.cm.winter_r
    elif isinstance(vertex_cmp, str):
        vertex_cmp = getattr(plt.cm, vertex_cmp)

    if erange is not None:
        emin_val, emax_val = erange
    else:
        emin_val, emax_val = -10, 10
    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin, vmax = None, None
    enorm = mpl.colors.Normalize(vmin=emin_val, vmax=emax_val)
    vnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if edge_attr is not None:
        graph.edge_attr = edge_attr.unsqueeze(-1)
    if node_attr is not None:
        graph.node_attr = node_attr.unsqueeze(-1)
    if isinstance(graph, pyg.data.Data):
        nx_graph = pyg.utils.to_networkx(graph, edge_attrs=["edge_attr"], node_attrs=["node_attr"], to_undirected=False)
    else:
        nx_graph = graph
        # set edge attributes
        for e,edge in enumerate(nx_graph.edges()):
            nx_graph.edges[edge]["edge_attr"] = edge_attr[e].item()
        # set node attributes
        for node in nx_graph.nodes():
            nx_graph.nodes[node]["node_attr"] = node_attr[node].item()


    pos = nx.kamada_kawai_layout(nx_graph)

    if node_attr is not None:
        node_color = [node[1]["node_attr"] for node in nx_graph.nodes(data=True)]
        if transform is not None:
            node_color = transform(np.array(node_color))
        if vmin is None:
            vmin = min(node_color)
        if vmax is None:
            med_color = sorted(node_color)[len(node_color) // 2]
            vmax = 2 * med_color

        other_nodes = [node for node in nx_graph.nodes() if node not in source_nodes and node != dest_nodes]
        other_node_colors = [node[1]["node_attr"] for node in nx_graph.nodes(data=True) if node[0] not in source_nodes and node[0] != dest_nodes]
        if transform is not None:
            other_node_colors = transform(np.array(other_node_colors))
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=other_nodes, node_size=600, alpha=0.5,
                               node_color=other_node_colors, cmap=vertex_cmp, vmin=vmin, vmax=vmax,
                               edgecolors="black")

        source_node_colors = [node[1]["node_attr"] for node in nx_graph.nodes(data=True) if node[0] in source_nodes]
        if transform is not None:
            source_node_colors = transform(np.array(source_node_colors))
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=source_nodes, node_size=2000, node_shape='^',
                               node_color=source_node_colors, cmap=vertex_cmp, vmin=vmin, vmax=vmax,
                               edgecolors="black")

        dest_node_colors = [node[1]["node_attr"] for node in nx_graph.nodes(data=True) if node[0] == dest_nodes]
        if transform is not None:
            dest_node_colors = transform(np.array(dest_node_colors))
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, nodelist=[dest_nodes], node_size=3000, node_shape='*',
                               node_color=dest_node_colors, cmap=vertex_cmp, vmin=vmin, vmax=vmax,
                               edgecolors="black")

        nx.draw_networkx_labels(nx_graph, pos, ax=ax, labels={i: f"{node_attr[i].item():.{1}f}" for i in range(node_attr.shape[0])})
    else:
        nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=100, edgecolors="black")
        nx.draw_networkx_labels(nx_graph, pos, ax=ax)
    if edge_attr is not None:
        edge_color = [np.array(edge[2]["edge_attr"]).mean() for edge in nx_graph.edges(data=True)]
        edges = nx.draw_networkx_edges(nx_graph, pos, ax=ax, connectionstyle="arc3,rad=0.2", edge_color=edge_color, edge_cmap=edge_cmp, edge_vmin=emin_val, edge_vmax=emax_val)
        labels = {edge: round(attr, 2) for edge, attr in nx.get_edge_attributes(nx_graph, "edge_attr").items()}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, ax=ax, connectionstyle="arc3,rad=0.2")
    else:
        nx.draw_networkx_edges(nx_graph, pos, ax=ax, connectionstyle="arc3,rad=0.2")
        edges = None
    if edges is not None:
        pc = mpl.collections.PatchCollection(edges, cmap=edge_cmp, norm=enorm)
        pc.set_array(edge_color)
    else:
        pc = None
    fig.suptitle(f"{title}\n{subtitle}", fontsize=32)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if pc is not None and show_colorbar:
        fig.colorbar(pc, ax=ax, orientation='vertical', label="Edge Weight")

    return fig, ax

def compute_lta(backlogs: np.ndarray):
    """
    Compute the long term average of the backlogs
    :param backlogs: backlog vector as a (N,) numpy array
    :return: long-term average backlogs as a (N,) numpy array
    """
    # compute cumalative sum of backlogs
    csum = np.cumsum(backlogs)
    divisor = np.arange(1, len(backlogs)+1)
    return np.divide(csum, divisor)