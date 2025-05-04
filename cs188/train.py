import os
import argparse
from typing import Any, Dict, List
import torch

import holdem.utils as utils
from holdem.utils import Logger
import holdem.environment as holdem_env
from holdem.environment import Env, num_actions, state_shape
from holdem.models import DQNAgent, RandomAgent, Agent, RuleBasedAgent, QLearningAgent


__env_config: Dict[str, Any] = {"num_players": 2, "seed": 3407}


def train(args):
    device = utils.get_device()
    utils.set_global_seed(__env_config["seed"])

    if args.load_checkpoint_path != "":
        agent = DQNAgent.from_checkpoint(
            checkpoint=torch.load(args.load_checkpoint_path)
        )
    else:
        agent = DQNAgent(
            num_actions=num_actions,
            state_shape=state_shape(__env_config["num_players"])[0],
            mlp_layers=[64, 64],
            device=device,
            save_path=args.log_dir,
            save_every=args.save_every,
        )
    # if args.load_checkpoint_path != "":
    #     agent = QLearningAgent(
    #         num_actions=num_actions,
    #         q_table_path=args.load_checkpoint_path,
    #     )
    # else:
    #     agent = QLearningAgent(
    #         num_actions=num_actions,
    #     )

    agents: List[Agent] = [agent]
    for _ in range(1, __env_config["num_players"]):
        agents.append(RuleBasedAgent(num_actions=num_actions))

    env = Env(**__env_config, agents=agents)

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            _, trajectories, payoffs = env.run(is_training=True)

            trajectories = holdem_env.transform_trajectory(
                trajectories, payoffs
            )

            for ts in trajectories[0]:
                agent.feed(ts)

            if episode % args.evaluate_every == 0:
                logger.log_perf(
                    episode,
                    holdem_env.evaluate_perf(
                        env,
                        args.num_eval_games,
                    )[0],
                )

        csv_path, fig_path = logger.csv_path, logger.fig_path

    utils.plot_curve(csv_path, fig_path, agent.__class__.__name__)

    save_path = os.path.join(args.log_dir, "model.pth")
    torch.save(agent, save_path)
    print("Model saved in", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=str,
        default="",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--num-eval-games",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="experiments/limit_holdem_dqn_result/",
    )

    parser.add_argument(
        "--load-checkpoint-path",
        type=str,
        default="",
    )

    parser.add_argument("--save-every", type=int, default=-1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
