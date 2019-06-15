from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
import torch

from a2c.sonic_environment import make_train_0#, make_train_1, make_train_2, make_train_3, make_train_4
from a2c.agent import Agent

num_steps = 1024
env = SubprocVecEnv([make_train_0, make_train_0, make_train_0, make_train_0, make_train_0, make_train_0, make_train_0, make_train_0, make_train_0, make_train_0])#, make_train_1, make_train_2, make_train_3, make_train_4])
agent = Agent(env.action_space.n)
agent.network.load_state_dict(torch.load('model.best.ckpt'))

max_reward = 0

state = env.reset().transpose(0, 3, 1, 2).astype(np.float32)
for current_episode in range(10000):
    state = env.reset().transpose(0, 3, 1, 2).astype(np.float32)

    log_probs = []
    rewards = []
    values = []
    dones = []
    entropies = 0

    for j in range(num_steps):
        action, log_action, entropy, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        next_state = next_state.transpose(0, 3, 1, 2).astype(np.float32)

        if done[0]:
            next_state = env.reset().transpose(0, 3, 1, 2).astype(np.float32)

        log_probs.append(log_action)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        entropies = entropies + entropy

    # add the last value so agent can learn from the experience
    with torch.no_grad():
        next_value = agent.get_value(next_state)
        values.append(next_value)

    print(f"episode: {current_episode} , reward: {np.array(rewards).mean()}")

    if max_reward < np.array(rewards).mean():
        max_reward = np.array(rewards).mean()
        torch.save(agent.network.state_dict(), 'model.best.ckpt')

    rewards = np.expand_dims(np.array(rewards), -1)
    dones = np.expand_dims(np.array(dones), -1)

    agent.learn(values, log_probs, entropies, dones, rewards)


