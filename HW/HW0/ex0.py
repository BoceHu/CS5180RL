import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    # TODO check if goal was reached
    goal_state = (10, 10)
    if state == goal_state:
        reward = 1
        new_state = reset()

        return new_state, reward

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)
    action_taken = action
    prob = np.random.rand()
    # check actions
    act_list = []
    for ac in Action:
        act_list.append(ac.value)
    # noise
    if 0 <= prob < 0.05:
        action_taken = Action(act_list[(action + 1) % 4])
        pass
    elif 0.05 <= prob < .1:
        action_taken = Action(act_list[(action - 1) % 4])

    # TODO calculate the next state and reward given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall

    # add surrounding walls
    for i in range(11):
        left_w = (-1, i)
        down_w = (i, -1)
        right_w = (11, i)
        up_w = (i, 11)
        walls += [left_w, down_w, right_w, up_w]

    next_state = tuple(origin + new for origin, new in zip(state, actions_to_dxdy(action_taken)))
    if next_state in walls:
        next_state = state
    reward = 0

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    print(f"The current state is: {state}, \
    please choose the actions from the follows: \
    LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3")
    action = input()
    act_list = []
    for ac in Action:
        act_list.append(ac.value)
    while True:
        if int(action) in act_list:
            action = Action(int(action))
            return action
        else:
            print("Please choose number from 【0, 1, 2, 3]")
            action = input()


# Q2
def agent(
        steps: int = 1000,
        trials: int = 1,
        policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    total_reward = []
    for t in range(trials):
        state = reset()
        i = 0
        reward_trial = 0
        cumulative_trial = []
        while i < steps:
            # TODO select action to take
            action = policy(state)
            # TODO take step in environment using simulate()
            state, reward = simulate(state, action)
            # TODO record the reward
            reward_trial += reward
            i += 1
            cumulative_trial.append(reward_trial)
        total_reward.append(cumulative_trial)

    return total_reward


# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    act_list = []
    for ac in Action:
        act_list.append(ac.value)
    action = Action(np.random.choice(act_list))

    return action


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return Action(0)


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    act_list = []
    for ac in Action:
        act_list.append(ac.value)

    key_up = []
    key_down = []
    key_left = []
    key_right = []
    for i in range(8):
        key_up.append([1, i])
        key_up.append([8, i])

    # for i in range(2, 7):
    #     # key_right.append([i, 1])
    #     # key_right.append([i, 8])
    #     pass
    #
    # for i in range(5, 10):
    #     key_up.append([9, i])
    #     key_up.append([10, i])

    # for i in range(2, 5):
    #     for j in range(2, 5):
    #         key_left.append([i, j])

    for i in range(5):
        key_down.append([i, 9])
        # key_down.append([i, 10])
        key_up.append([i, 6])
        key_up.append([i, 7])
        # key_up.append([i, 0])
        key_right.append([0, i])

    # for i in range(11):
    #     key_right.append([6, i])
    #     # key_right.append([7, i])

    for i in range(4):
        key_left.append([9, i])
        # key_left.append([10, i])
    key_right.append([9, 10])
    state = list(state)
    if state in key_up:
        action = Action(3)
    elif state in key_left:
        action = Action(0)
    elif state in key_down:
        action = Action(1)
    elif state in key_right:
        action = Action(2)
    else:
        action = Action(np.random.choice(act_list))
    return action


def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question
    agent_q3_r = agent(10000, 10, random_policy)
    for i in range(10):
        plt.plot(agent_q3_r[i], ':')

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.plot(np.average(np.array(agent_q3_r), 0), color='black', label='Random Policy', linewidth=2)
    plt.legend()
    plt.show()

    # Q4 PLOT
    reward_q4_better = agent(10000, 10, better_policy)
    reward_q4_rand = agent(10000, 10, random_policy)
    reward_q4_worse = agent(10000, 10, worse_policy)

    for i in range(10):
        plt.plot(reward_q4_better[i], ':')  # better policy
        plt.plot(reward_q4_rand[i], ':')  # random policy
        plt.plot(reward_q4_worse[i], ':')  # worse policy

    # Average rewards
    plt.plot(np.average(np.array(reward_q4_worse), 0), color='green', label='Worse Policy', linewidth=2)
    plt.plot(np.average(np.array(reward_q4_rand), 0), color='black', label='Random Policy', linewidth=2)
    plt.plot(np.average(np.array(reward_q4_better), 0), color='blue', label='Better Policy', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
