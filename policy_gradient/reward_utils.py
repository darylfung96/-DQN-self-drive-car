import numpy as np


def normalize_rewards(episode_rewards, gamma):
    accumulated_rewards = 0
    discounted_rewards = np.zeros_like(episode_rewards)
    for i in reversed(range(len(episode_rewards))):
        accumulated_rewards = accumulated_rewards * gamma + episode_rewards[i]
        discounted_rewards[i] = accumulated_rewards

    normalized_rewards = discounted_rewards - np.mean(discounted_rewards) / np.std(discounted_rewards)
    return normalized_rewards
