import json
import os
import platform
import argparse
from evaluation.eval_training_model_mp import evaluate_model as evaluate_model


import warnings
warnings.filterwarnings("ignore")

opsys = platform.system()
if opsys == "Windows":
    PROJECT_DIR = "C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\"
else:
    PROJECT_DIR = "/home/jwigmore/PycharmProjects/GDRL4Nets"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type = str, required = True)
    parser.add_argument("--model_name", type = str, required = True)
    args = parser.parse_args()

    run_path = args.run_path
    model_name = args.model_name

    eval_results = evaluate_model(run_path, model_name)

    print(f"Eval Finished \n Set Average Mean Norm Score {eval_results["Set_Average"]["mean_norm_score"]}")
