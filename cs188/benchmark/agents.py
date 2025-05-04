from holdem.models import DQNAgent, RandomAgent, RuleBasedAgent, QLearningAgent, Agent
import torch
from holdem.environment import num_actions as NUM_ACTIONS

def create_ql_agent(**kargs) -> QLearningAgent:
    return QLearningAgent(**kargs)
 
def create_dqn_agent(**kargs) -> DQNAgent:
    checkpoint = kargs.get("checkpoint", None)
    if checkpoint:
        return DQNAgent.from_checkpoint(checkpoint=torch.load(checkpoint))
    else:
        print("Warning: creating new DQN agent")
        agent = DQNAgent(**kargs)
        return agent


def create_random_agent(**kargs) -> RandomAgent:
    return RandomAgent(**kargs)


def create_rulebased_agent(**kargs) -> RuleBasedAgent:
    return RuleBasedAgent(**kargs)


def create_agent(type="dqn", **kargs) -> Agent:
    match type:
        case "dqn":
            return create_dqn_agent(**kargs)
        case "random":
            return create_random_agent(**kargs)
        case "rulebased":
            return create_rulebased_agent(**kargs)
        case "ql":
            return create_ql_agent(**kargs)
        case other:
            raise ValueError(f"Unknown agent type: {type}")
