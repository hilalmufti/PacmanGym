from collections import defaultdict

import gymnasium as gym
from gymnasium import Env
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange


def policy(obs, qs, eps, pred, act, rand):
    if pred(eps):
        return rand(obs, qs)
    else:
        return act(obs, qs)


def policy_eps_greedy(obs, qs, eps, act, rand):
    return policy(obs, qs, eps, lambda eps: np.random.rand() < eps, act, rand)


def decay(eps: float, dec: float, flr: float):
    return max(flr, eps - dec)


def loss(x2, x1, bias=0):
    return (x2 - x1) + bias


def loss_td(q_new, q_old, reward, discount_factor):
    return loss(discount_factor * q_new, q_old, reward)


def opt(x, obj, lr):
    return x - lr * obj


def opt_td(q, td, lr):
    return opt(q, td, -lr)


def repeatedly(fn, it):
    return [fn(i) for i in (range(it) if isinstance(it, int) else it)]


def step(obs, env, agent):
    action = agent.get_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    agent.update(obs, action, reward, terminated, next_obs)

    return next_obs, terminated or truncated, info


def play(env, agent):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, done, _ = step(obs, env, agent)


def train_step(env, agent):
    play(env, agent)
    agent.decay_epsilon()


def train(env, agent, n_episodes):
    repeatedly(lambda _: train_step(env, agent), trange(n_episodes))


def visualize(env, agent):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
    axs[1].set_title("Episode Length")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")

    axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
    axs[2].set_title("Training Error")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Temporal Difference")

    plt.tight_layout()
    plt.show()


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
        return policy_eps_greedy(
            obs,
            self.q_values,
            self.epsilon,
            lambda obs, qs: int(np.argmax(qs[obs])),
            lambda obs, qs: self.env.action_space.sample(),
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

        l = loss_td(q_new, q_old, reward, self.discount_factor)
        self.q_values[obs][action] = opt_td(q_old, l, self.lr)

        self.training_error.append(l)

    def decay_epsilon(self):
        self.epsilon = decay(self.epsilon, self.epsilon_decay, self.final_epsilon)

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

    env = gym.make("Blackjack-v1", sab=False)

    agent = BlackjackAgent(
        env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    train(env, agent, n_episodes)
    visualize(env, agent)


if __name__ == "__main__":
    main()
