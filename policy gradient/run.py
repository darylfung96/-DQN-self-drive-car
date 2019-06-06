import gym
import numpy as np
import torch

from ai import Network

max_episode = 1000
num_steps = 500
gamma = 0.95

environment = gym.make('CartPole-v0')
input_shape = environment.observation_space.shape
output_shape = environment.action_space.n

network = Network(input_shape[0], output_shape)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)


def normalize_rewards(episode_rewards):
    accumulated_rewards = 0
    discounted_rewards = np.zeros_like(episode_rewards)
    for i in reversed(range(len(episode_rewards))):
        accumulated_rewards = accumulated_rewards * gamma + episode_rewards[i]
        discounted_rewards[i] = accumulated_rewards

    normalized_rewards = discounted_rewards - np.mean(discounted_rewards) / np.std(discounted_rewards)
    return normalized_rewards


def learn(states, actions, discounted_rewards):
    action_probs = network(states)

    discounted_rewards = np.expand_dims(discounted_rewards, -1).astype(np.float32)
    actions = np.expand_dims(actions, -1)
    actions = torch.from_numpy(actions)
    action_probs = torch.gather(action_probs, dim=1, index=actions)

    loss = -torch.log(action_probs) * torch.from_numpy(discounted_rewards)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

max_reward = 0
for current_episode in range(max_episode):

    episode_rewards = []
    states = []
    actions = []
    state = environment.reset()
    states.append(state)
    state = np.expand_dims(state, 0)
    for current_step in range(num_steps):
        action_prob = network(state)
        action = np.random.choice(range(action_prob.shape[1]), p=action_prob.data.numpy().ravel())
        actions.append(action)

        next_state, reward, done, _ = environment.step(action)
        if not done: states.append(next_state)
        next_state = np.expand_dims(next_state, 0)
        state = next_state

        episode_rewards.append(reward)

        if done:
            discounted_rewards = normalize_rewards(episode_rewards)
            states = np.array(states)
            actions = np.array(actions)
            learn(states, actions, discounted_rewards)
            if sum(episode_rewards) > max_reward:
                max_reward = sum(episode_rewards)
            print(f'episode {current_episode}, reward: {sum(episode_rewards)}')
            break


print(f"max reward: {max_reward}")