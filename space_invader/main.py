import numpy as np
import torch
import retro
import random

import torch.optim as optim

from memory import Memory
from dqn import DQN
from preprocess import stack_frames

# initialize environment
environment = retro.make(game='SpaceInvaders-Atari2600')
actions = np.eye(environment.action_space.n)
stack_frame_size = 4

# hyperparameters
gamma = 0.99
learning_rate = 0.0025
observation_size = [110, 84, 4]  # 4 refres to the number of stacked frame size
num_actions = environment.action_space.n

epsilon = 1.
min_epsilon = 0.01
epsilon_decay = 0.00002

batch_size = 64
pretrain_length = batch_size
max_steps = 5000
total_episode = 100
save_iter = 1000

is_render = False

# intialize agent
dqn = DQN(4, num_actions)
memory = Memory(10000)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)

try:
    model_state_dict = torch.load('model/model.best.ckpt(1)')
    dqn.load_state_dict(model_state_dict)
except FileNotFoundError:
    pass

# deal with empty memory problem
def pretrain_memory():
    state = environment.reset()
    state, stacked_frames = stack_frames(None, state, True)
    for i in range(pretrain_length):
        action_index = random.randint(0, environment.action_space.n-1)
        action = actions[action_index]
        next_state, reward, done, _ = environment.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_done=done)

        if done:
            next_state = np.zeros(environment.observation_space.shape)
            memory.add((state, action_index, next_state, reward, done))
            state = environment.reset()
            state, stacked_frames = stack_frames(None, state, True)
        else:
            memory.add((state, action_index, next_state, reward, done))
            state = next_state


def act(state):
    global epsilon
    if random.random() < epsilon:
        action_index = random.randint(0, environment.action_space.n - 1)
        action = actions[action_index]
        return action, action_index
    else:
        state = np.expand_dims(state, 0).astype(np.float32)
        action = dqn(state)
        best_action_index = action.argmax(1).item()
        best_action = actions[best_action_index]

    # decay exploration
    epsilon = min(min_epsilon, epsilon - epsilon_decay)

    return best_action, best_action_index


def learn():
    samples = memory.sample(batch_size)

    state, action_index, next_state, reward, done = list(zip(*samples))

    state = np.array(state)
    reward = np.expand_dims(np.array(reward), -1).astype(np.float32)
    next_state = np.array(next_state)
    action_index = np.expand_dims(np.array(action_index), -1)
    done = np.expand_dims(np.array(done), -1).astype(np.float32)

    with torch.no_grad():
        next_q_value, _ = dqn(next_state).max(1)
        target = reward + gamma * np.expand_dims(next_q_value.data.numpy(), -1) * (1 - done)

    q_value = dqn(state)
    q_value = torch.gather(q_value, dim=1, index=torch.from_numpy(action_index))
    optimizer.zero_grad()
    loss = (q_value - torch.from_numpy(target)).pow(2).mean()
    loss.backward()
    optimizer.step()

count = 0
max_rewards = 0


def train():
    global count, max_rewards
    for i in range(total_episode):
        state = environment.reset()
        state, stacked_frames = stack_frames(None, state, True)
        total_rewards = 0

        for _ in range(max_steps):
            action, action_index = act(state)
            next_state, reward, done, _ = environment.step(action)
            environment.render()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            total_rewards += reward

            if done:
                print(f"episode: {i}, total_rewards: {total_rewards}")
                next_state = environment.reset()
                next_state, stacked_frames = stack_frames(None, next_state, True)
                memory.add((state, action_index, next_state, reward, done))

                if total_rewards > max_rewards:
                    max_rewards = total_rewards
                    torch.save(dqn.state_dict(), 'model/model.best.ckpt(1)')

                break
            else:
                memory.add((state, action_index, next_state, reward, done))

            # learn()
            count += 1

            if count % save_iter == 0:
                torch.save(dqn.state_dict(), 'model/model.latest.ckpt')


if __name__ == '__main__':
    pretrain_memory()
    train()


