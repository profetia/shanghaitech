import copy
from typing import List
import numpy as np
from .agent import Agent
from multiprocessing import Pool
from ..environment.hand import Hand
from ..environment.card import index_map


__weights = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def sample_future(card_indexes: List[int]):
    remain_indexes = list(set([i for i in range(52)]) - set(card_indexes))
    while len(card_indexes) < 7:
        card_indexes.append(
            remain_indexes.pop(np.random.randint(0, len(remain_indexes)))
        )

    cards: List[str] = list(map(lambda x: index_map[x], card_indexes))

    return __weights[Hand(cards).evaluate().value]


class RuleBasedAgent(Agent):
    def __init__(self, num_actions,iter_num=1000):
        self.num_actions = num_actions
        self.thread_pool = Pool()
        self.iter_num=iter_num

    def step(self, state):
        # state space
        # Index	Meaning
        # 0~12	Spade A ~ Spade K
        # 13~25	Heart A ~ Heart K
        # 26~38	Diamond A ~ Diamond K
        # 39~51	Club A ~ Club K
        # 52~56	Raise number in round 1
        # 57~61	Raise number in round 2
        # 62~66	Raise number in round 3
        # 67~71	Raise number in round 4
        # action space
        # Action ID	Action
        # 0	Call
        # 1	Raise
        # 2	Fold
        # 3	Check
        cards = []
        for i in range(52):
            cards.append(i) if state["obs"][i] == 1 else None
        num_cards = len(cards)
        action = -1
        if num_cards == 2:
            # one pair, raise
            if cards[0] % 13 == cards[1] % 13:
                action = 1
            else:
                if 0 in state["legal_actions"]:
                    action = 0
                else:
                    action = 1
        if num_cards == 7:
            level = (
                Hand(list(map(lambda x: index_map[x], cards))).evaluate().value
            )
            if level == 1:
                action = 2
            elif level <= 3:
                if 3 in state["legal_actions"]:
                    action = 3
                else:
                    action = 2
            elif level <= 5:
                if 0 in state["legal_actions"]:
                    action = 0
                else:
                    action = 1
            elif level <= 7:
                action = 1

        score = self.calculate_sum(cards)
        if num_cards == 5:
            if score < 4:
                action = 2
            elif score < 15:
                if 3 in state["legal_actions"]:
                    action = 3
                else:
                    action = 2
            elif score < 25:
                if 0 in state["legal_actions"]:
                    action = 0
                else:
                    action = 1
            else:
                action = 1
        elif num_cards == 6:
            if score < 3:
                action = 2
            elif score < 10:
                if 3 in state["legal_actions"]:
                    action = 3
                else:
                    action = 2
            elif score < 20:
                if 0 in state["legal_actions"]:
                    action = 0
                else:
                    action = 1
            else:
                action = 1
        if action == -1:
            action = np.random.choice(list(state["legal_actions"].keys()))
        return action

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

    def calculate_sum(self, cards):
        sample_result = self.thread_pool.starmap(
            sample_future, [(copy.deepcopy(cards),) for _ in range(self.iter_num)]
        )
        return sum(sample_result) / self.iter_num
