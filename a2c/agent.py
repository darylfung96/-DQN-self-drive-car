import torch
import numpy as np
import torch.optim as optim

from a2c.network import ActorCritic

gamma = 0.95
lam = 0.95
batch_size = 1024


class Agent(object):
    def __init__(self, number_actions):
        self.network = ActorCritic(number_actions)
        self.optimizer = optim.Adam(self.network.parameters())

    def select_action(self, state):
        act_dist, value = self.network(state)
        action = act_dist.sample()
        entropy = act_dist.entropy().mean()
        log_action = act_dist.log_prob(action).reshape(state.shape[0], -1)
        return action, log_action, entropy, value

    def get_value(self, state):
        _, value = self.network(state)
        return value

    def calculate_returns(self, values, dones, rewards):
        returns = []
        lastgae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i+1].data.numpy() * (1 - dones[i]) - values[i].data.numpy()
            R = lastgae = delta + gamma * lam * (1 - dones[i]) * lastgae
            returns.insert(0, R.astype(np.float32))
        return torch.from_numpy(np.array(returns))

    def learn(self, values, log_actions, entropies, dones, rewards):
        returns = self.calculate_returns(values, dones, rewards)

        # values[:-1] because the last value does not have a target
        # we append the last next_state value to the values array just so that we can calculate the return
        # easily
        values = values[:-1]
        batches = returns.shape[0] // batch_size

        for batch_index in range(batches):
            start_index = batch_index * batch_size
            end_index = (batch_index+1) * batch_size

            log_actions_batch = torch.stack(log_actions[start_index:end_index])
            values_batch = values[start_index:end_index]
            returns_batch = returns[start_index:end_index]

            actor_loss = -(log_actions_batch * returns_batch).mean()
            advantage = returns - torch.stack(values_batch)
            value_loss = advantage.pow(2).mean()

            self.optimizer.zero_grad()
            total_loss = actor_loss + 0.5 * value_loss - 0.001 * entropies
            total_loss.backward()
            self.optimizer.step()


