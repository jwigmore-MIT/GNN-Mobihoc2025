"""
From a context set folder, compute the metric stats for all metrics in each context
"""
from utils import *
import os
import json
from create_random_network_ts_bp import compute_metric_stats

if __name__ == "__main__":
    # Load context set
    context_set_dir = "experiments\\GNNBiasedBackpressureDevelopment\\context_sets\\barabasi_albert_20_context_set_3_19b"
    context_set = os.listdir(os.path.join(PROJECT_DIR, context_set_dir))
    metrics = []
    for context in context_set:
        if "Env" not in context:
            continue
        with open(os.path.join(PROJECT_DIR, context_set_dir, context), 'r') as f:

            context_config = json.load(f)
        metrics.append(context_config['metrics'])

    context_set_stats = compute_metric_stats(metrics)

    for key in context_set_stats:
        print(key, context_set_stats[key])

    # Save context set stats
    with open(os.path.join(PROJECT_DIR, context_set_dir, "context_set_stats.json"), 'w') as f:
        json.dump(context_set_stats, f)