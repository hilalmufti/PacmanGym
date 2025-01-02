from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium import Env
from tqdm import tqdm


def get_action(obs, q_values, sampler, epsilon):
    if np.random.random() < epsilon:
        return sampler()
    else:
        return int(np.argmax(q_values[obs]))

def decay_epsilon(epsilon: float, epsilon_decay: float, final_epsilon: float):
    return max(final_epsilon, epsilon - epsilon_decay)


def td(q_new, q_old, reward, discount_factor):
    return reward + (discount_factor * q_new) - q_old


def step_td(q, td, lr):
    assert not callable(td)
    return q + lr * td


def update_q(
    q_new, q_old, reward: float, terminated: bool, discount_factor: float, lr: float
):
    td_error = td(q_new, q_old, reward, discount_factor)
    return step_td(q_old, td_error, lr), td_error


def pprint_env(env, n=0):
    name = env.spec.id
    print(f"-- {name} --")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    if hasattr(env, "reward_range"):
        print(f"Reward Range: {env.reward_range}")
    print(f"Metadata: {env.metadata}")
    print(f"Spec: {env.spec}")

    for _ in range(n):
        print()


class BlackjackAgent:
    def __init__(
        self,
        env: Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        return get_action(
            obs, self.q_values, self.env.action_space.sample, self.epsilon
        )

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        q_new = (not terminated) * np.max(self.q_values[next_obs])
        q_old = self.q_values[obs][action]

        self.q_values[obs][action], td_error = update_q(
            q_new, q_old, reward, terminated, self.discount_factor, self.lr
        )
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = decay_epsilon(
            self.epsilon, self.epsilon_decay, self.final_epsilon
        )

    def __repr__(self):
        return (
            f"BlackjackAgent(env={self.env}, learning_rate={self.lr}, "
            f"q_values={self.q_values}, initial_epsilon={self.epsilon}, "
            f"epsilon_decay={self.epsilon_decay}, final_epsilon={self.final_epsilon}, "
            f"discount_factor={self.discount_factor})"
        )


def main():
    learning_rate = 0.01
    n_episodes = 100_000
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (n_episodes / 2)  # 0.00002
    final_epsilon = 0.1
    discount_factor = 0.95

    env = gym.make("Blackjack-v1", sab=False)
    pprint_env(env, 1)

    agent = BlackjackAgent(
        env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon
    )
    print(agent)
    print()

    for i in range(70000):
        if i % 7000 == 0:
            print(agent.get_action((1, 1, False)), agent.epsilon)
        agent.decay_epsilon()


if __name__ == "__main__":
    main()
