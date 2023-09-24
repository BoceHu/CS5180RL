import numpy as np

from env import BanditEnv
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from agent import EpsilonGreedy


def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # TODO
    reward_all = list()
    for i in range(k):
        reward_each = list()
        for j in range(num_samples):
            reward = env.step(i)
            reward_each.append(reward)

        reward_all.append(reward_each)

    plt.violinplot(dataset=reward_all, showmedians=True, showextrema=False)
    plt.xlabel("Action")
    plt.ylabel("Reward Distribution")
    plt.show()


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k)
    epsilon_0 = EpsilonGreedy(k=10, init=0, epsilon=0)
    epsilon_001 = EpsilonGreedy(k=10, init=0, epsilon=0.01)
    epsilon_01 = EpsilonGreedy(k=10, init=0, epsilon=0.1)
    agents = [epsilon_0, epsilon_001, epsilon_01]

    # Loop over trials
    R = list()
    R_optimal = list()
    upper_list = list()
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        upper_bound = env.means.max()
        upper_list.append(upper_bound)
        r_trail = list()
        r_trail_optimal = list()
        for agent in agents:
            agent.reset()
            r_agent = list()
            r_agent_optimal = list()
            # TODO For each trial, perform specified number of steps for each type of agent
            for step in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                r_agent.append(reward)
                if action == np.argmax(env.means):
                    r_agent_optimal.append(1)
                else:
                    r_agent_optimal.append(0)
            r_trail.append(r_agent)
            r_trail_optimal.append(r_agent_optimal)
        R.append(r_trail)
        R_optimal.append(r_trail_optimal)

    R_avg = np.average(R, axis=0)
    R_optimal_avg = np.average(R_optimal, axis=0)
    R_std_avg = np.std(R, axis=0)

    upper_avg = np.average(upper_list)

    e_0 = R_avg[0]
    e_001 = R_avg[1]
    e_01 = R_avg[2]

    e_0_optimal = R_optimal_avg[0]
    e_001_optimal = R_optimal_avg[1]
    e_01_optimal = R_optimal_avg[2]

    ste = R_std_avg / np.sqrt(trials)
    up_ste = upper_avg / np.sqrt(trials)
    e_0_h = (e_0 + 1.96 * ste[0])
    e_0_l = (e_0 - 1.96 * ste[0])

    e_001_h = (e_001 + 1.96 * ste[1])
    e_001_l = (e_001 - 1.96 * ste[1])

    e_01_h = (e_01 + 1.96 * ste[2])
    e_01_l = (e_01 - 1.96 * ste[2])
    upper_h = (upper_avg + 1.96 * up_ste)
    upper_l = (upper_avg - 1.96 * up_ste)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18), dpi=200)

    def to_percent(y, position):
        return f'{100 * y:.0f}%'

    formatter = FuncFormatter(to_percent)

    ax2.yaxis.set_major_formatter(formatter)
    ax1.plot(e_0, label="ε = 0 (greedy)", c='palegreen')
    ax1.plot(e_001, label="ε = 0.01", c='tomato')
    ax1.plot(e_01, label="ε = 0.1", c='skyblue')

    ax2.plot(e_0_optimal, label="ε = 0 (greedy)", c='palegreen')
    ax2.plot(e_001_optimal, label="ε = 0.01", c='tomato')
    ax2.plot(e_01_optimal, label="ε = 0.1", c='skyblue')
    x = np.arange(steps)
    ax1.axhline(upper_avg, linestyle='--')
    ax1.fill_between(x, e_0_l, e_0_h, alpha=0.3, color='palegreen')
    ax1.fill_between(x, e_001_l, e_001_h, alpha=0.3, color='tomato')
    ax1.fill_between(x, e_01_l, e_01_h, alpha=0.3, color='skyblue')
    ax1.fill_between(x, upper_l, upper_h, alpha=0.3)

    ax1.set_xlabel("Steps", fontsize=23)
    ax2.set_xlabel("Steps", fontsize=23)
    ax1.set_xticks(range(0, 2001, 500))
    ax2.set_xticks(range(0, 2001, 500))
    ax1.set_ylabel("Average Reward", fontsize=23)
    ax2.set_ylabel("Optimal Action", fontsize=23)

    ax1.legend(fontsize=20, loc=4)
    ax2.legend(fontsize=20, loc=4)
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    plt.show()


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = None
    agents = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()

        # TODO For each trial, perform specified number of steps for each type of agent

    pass


def main():
    # TODO run code for all questions
    # q4(10, 2000)
    q6(10, 2000, 2000)


if __name__ == "__main__":
    # main()
    action = np.where(np.array([0, 5, 4, 0, 0]) == 0)[0]
    print(action)
