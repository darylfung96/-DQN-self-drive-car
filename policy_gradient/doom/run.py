from vizdoom import DoomGame
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from skimage import transform

from policy_gradient.ai import ConvNetwork
from policy_gradient.reward_utils import normalize_rewards

max_episodes = 1000
max_steps = 1000
learning_rate = 0.001
stacked_frame_size = 4
gamma = 0.95


def create_environment():
    game = DoomGame()

    # Load the correct configuration
    game.load_config("health_gathering.cfg")

    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path("health_gathering.wad")

    game.init()

    # Here our possible actions
    # [[1,0,0],[0,1,0],[0,0,1]]
    possible_actions = np.identity(3, dtype=int).tolist()

    return game, possible_actions


def preprocess_state(state):
    state = state[80:, :]  # remove roof, cotains no info
    normalized_state = state / 255.0
    preprocessed_state = transform.resize(normalized_state, (64, 64))
    return preprocessed_state


def stack_frames(stacked_frames, new_frame, is_new_episode):
    if stacked_frames is None:
        stacked_frames = [np.zeros((80, 80)) for _ in range(stacked_frame_size)]

    new_frame = preprocess_state(new_frame)

    if is_new_episode:
        stacked_frames = [np.zeros((80, 80)) for _ in range(stacked_frame_size)]
        stacked_frames = deque(stacked_frames, maxlen=stacked_frame_size)

        for _ in range(stacked_frame_size):
            stacked_frames.append(new_frame)
    else:
        stacked_frames.append(new_frame)

    stacked_states = np.stack(stacked_frames, axis=0).astype(np.float32)
    return stacked_states, stacked_frames


def learn(states, actions, rewards):
    action_prob = dqn(states)

    action_prob = torch.gather(action_prob, dim=-1, index=torch.from_numpy(actions))
    action_log_prob = torch.log(action_prob)

    loss = -action_log_prob * torch.from_numpy(rewards)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


game, actions = create_environment()
dqn = ConvNetwork([80, 80], 3)
optimizer = optim.Adam(dqn.parameters())

states_history = []
actions_history = []
rewards_history = []

for eps in range(max_episodes):
    game.new_episode()

    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(None, state, True)

    for step in range(max_steps):
        action_prob = dqn(np.expand_dims(state, 0))
        action = np.random.choice(range(action_prob.shape[1]), p=action_prob.data.numpy().ravel())
        reward = game.make_action(actions[action])
        done = game.is_episode_finished()

        states_history.append(state)
        actions_history.append(action)
        rewards_history.append(reward)

        if done:
            game.new_episode()
            if len(rewards_history) > 2000:
                discounted_rewards = normalize_rewards(rewards_history, gamma)
                states_history = np.array(states_history)
                actions_history = np.expand_dims(np.array(actions_history), -1)
                discounted_rewards = np.expand_dims(discounted_rewards, -1).astype(np.float32)
                learn(states_history, actions_history, discounted_rewards)
                # reset histories or else will not be on-policy anymore
                states_history = []
                actions_history = []
                rewards_history = []
                break

        else:
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
