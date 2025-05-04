import numpy as np
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, state):
        return np.random.choice(list(state["legal_actions"].keys()))

    def eval_step(self, state):
        probs = [0 for _ in range(self.num_actions)]
        for i in state["legal_actions"]:
            probs[i] = 1 / len(state["legal_actions"])

        info = {}
        info["probs"] = {
            state["raw_legal_actions"][i]: probs[
                list(state["legal_actions"].keys())[i]
            ]
            for i in range(len(state["legal_actions"]))
        }

        return self.step(state), info
