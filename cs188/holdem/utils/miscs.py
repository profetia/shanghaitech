import torch
from torch import device
import os
import csv
from typing import List
import matplotlib.pyplot as plt


def get_device() -> device:
    if torch.cuda.is_available():
        result = device("cuda")
    else:
        result = device("cpu")
    return result


def plot_curve(csv_path, fig_path, agent_name):
    with open(csv_path, "r") as f:
        reader: csv.DictReader = csv.DictReader(f)

        xs: List[int] = []
        ys: List[float] = []
        for row in reader:
            xs.append(int(row["episode"]))
            ys.append(float(row["reward"]))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=agent_name)
        ax.set(xlabel="episode", ylabel="reward")
        ax.legend()
        ax.grid()

        save_dir: str = os.path.dirname(fig_path)
        os.makedirs(save_dir, exist_ok=True)

        fig.savefig(fig_path)
