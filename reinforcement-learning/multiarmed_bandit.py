import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MultiArmedBanditEnv(gym.Env):
    def __init__(self, bandits):
        self.bandits = bandits
        self.reset()

    def step(self, action):
        p = self.bandits[action]
        reward = 1 if random.random() <= p else -1
        self.state[action].append(reward)
        done = False
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = defaultdict(list)
        info = {}
        return self.state, info

    def render(self):
        print("\n--- Bandit Summary ---")
        for i, rewards in self.state.items():
            print(f'Bandit {i}: Pulled {len(rewards)} times | Avg return: {np.mean(rewards):.2f}')
        print(f'Total Trials: {sum(len(v) for v in self.state.values())}')
        print(f'Total Returns: {sum(sum(v) for v in self.state.values())}')


def get_bandit_env(bandits=None):
    if bandits is None:
        bandits = [.45, .45, .4, .6, .4]
    return MultiArmedBanditEnv(bandits)


def epsilon_greedy_action(Q, counts, epsilon, n_bandits):
    if random.random() < epsilon:
        return random.randint(0, n_bandits - 1)  # explore
    else:
        return max(range(n_bandits), key=lambda x: Q[x])  # exploit


def plot_results(rewards, average_rewards, pulls):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.7)
    plt.title("Rewards Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.subplot(1, 3, 2)
    plt.plot(average_rewards)
    plt.title("Average Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")

    plt.subplot(1, 3, 3)
    plt.bar(range(len(pulls)), pulls)
    plt.title("Bandit Pull Counts")
    plt.xlabel("Bandit")
    plt.ylabel("Pull Count")

    plt.tight_layout()
    plt.show()


def simulate_bandits(steps=1000, epsilon=0.1):
    env = get_bandit_env()
    state = env.reset()
    n_bandits = len(env.bandits)

    Q = [0.0] * n_bandits  # estimated values
    counts = [0] * n_bandits  # action counts

    rewards = []
    avg_rewards = []
    pulls = [0] * n_bandits

    for t in range(1, steps + 1):
        action = epsilon_greedy_action(Q, counts, epsilon, n_bandits)
        state, reward, done,truncated, info = env.step(action)

        # update estimates
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        rewards.append(reward)
        avg_rewards.append(np.mean(rewards))
        pulls[action] += 1

    env.render()
    plot_results(rewards, avg_rewards, pulls)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    simulate_bandits(steps=1000, epsilon=0.1)
