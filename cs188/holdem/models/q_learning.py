# This code is derived from Berkeley CS188 Pacman Project 6: Reinforcement Learning
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import numpy as np
from .agent import Agent
import json
import itertools

class QLearningAgent(Agent):
    def __init__(self, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_path=None):
        self.num_actions = num_actions
        self.q_table_path = q_table_path
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_count = 0
        if self.q_table_path:
            self.load(self.q_table_path)
    
    def getQValue(self, state, action):
        state = tuple(state)
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def computeValueFromQValues(self, state):
        tmp = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            tmp[action] = self.getQValue(state, action)
        return tmp.max()
    
    def computeActionFromQValues(self, state):
        tmp = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            tmp[action] = self.getQValue(state['obs'], action)
        if tmp.max() == 0:
            return np.random.choice(list(state["legal_actions"].keys()))
        return tmp.argmax()

    def step(self, state):
        legalActions = state["legal_actions"]
        action = None
        if not legalActions:
            return None
        action = self.computeActionFromQValues(state) if np.random.rand() > self.epsilon else np.random.choice(list(legalActions.keys()))
        return action
    
    def update(self, state, action, nextState, reward):
        legalActions = nextState["legal_actions"]
        state = state["obs"]
        nextState = nextState["obs"]
        if not legalActions:
            return None
        q_value = self.getQValue(state, action)
        next_max_q_value = self.computeValueFromQValues(nextState)
        update = reward + self.gamma * next_max_q_value
        self.q_table[(tuple(state), action)] += self.alpha * update


    def eval_step(self, state):
        self.step_count += 1
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
        action = self.step(state)
        # if self.step_count % 100 == 0:
        #     self.save(self.q_table_path)
        return action, info
    
    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.update(state, action, next_state, reward)

    def save(self, filename):
        print("save q_table at iteration", self.step_count)
        print("q_table size", len(self.q_table))
        if self.step_count == 5000:
            raise Exception("stop")
        np.save(filename, self.q_table)
    
    def load(self, filename):
        print("load q_table")
        self.q_table = np.load(filename, allow_pickle=True).item()
        # self.q_table = dict(itertools.islice(self.q_table.items(), 10)) 
        print("q_table size", len(self.q_table))