import random

from poutyne import Model
from copy import deepcopy  # NEW

import numpy as np
import gym
import torch


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size
        self.__buffer = []

    def __len__(self):
      return len(self.__buffer)

    def store(self, element):
        '''
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        '''
        self.__buffer.append(element)
        if len(self.__buffer) > self.__buffer_size:
            del self.__buffer[0]

    def get_batch(self, batch_size):
        '''
        Returns a list of batch_size elements from the buffer.
        '''        
        return random.choices(self.__buffer, k=batch_size)


class DQN(Model):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def get_action(self, state, epsilon):
        '''
        Returns the selected action according to an epsilon-greedy policy.
        '''
        if(np.random.random() < epsilon):
            return np.random.choice(self.actions)
        
        return np.argmax(self.predict_on_batch(state))

    def soft_update(self, other, tau):
        '''
        Code for the soft update between a target network (self) and
        a source network (other).

        The weights are updated according to the rule in the assignment.
        '''
        new_weights = {}

        own_weights = self.get_weight_copies()
        other_weights = other.get_weight_copies()

        for k in own_weights:
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

        self.set_weights(new_weights)


class NNModel(torch.nn.Module):
    '''
    Neural Network with 3 hidden layers of hidden dimension 64.
    '''
    def __init__(self, in_dim, out_dim, n_hidden_layers=3, hidden_dim=64):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.fa = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fa(x)


def format_batch(batch, target_network, gamma):
    '''
    Input : 
        - batch, a list of n=batch_size elements from the replay buffer
        - target_network, the target network to compute the one-step lookahead target
        - gamma, the discount factor

    Returns :
        - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
        - (actions, targets) : where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.
    '''
    states, actions, rewards, states_, are_terminal_states = zip(*batch)
    states = np.vstack(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    states_ = np.vstack(states_)
    are_terminal_states = np.array(are_terminal_states)
    
    q_theta_ = target_network.predict_on_batch(states_)
    targets = rewards + gamma * np.max(q_theta_, axis=1) * (1-are_terminal_states)
    
    return states, (actions, targets.astype(np.float32))

def dqn_loss(y_pred, y_target):
    '''
    Input :
        - y_pred, (batch_size, n_actions) Tensor outputted by the network
        - y_target = (actions, targets), where actions and targets both
                      have the shape (batch_size, ). Actions are the 
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.

    Returns :
        - The DQN loss 
    '''
    actions, targets = y_target
    q_thetas = y_pred.gather(1, actions.type(torch.int64).unsqueeze(-1)).squeeze()

    return torch.nn.functional.mse_loss(targets, q_thetas)

def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW

def run(batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate,  epsilon=1.0, epsilon_min=0.01, epochs=600, T=1000, start_replay=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment = gym.make("LunarLander-v2")
    set_random_seed(environment, seed)

    observation_space = environment.observation_space
    action_space = environment.action_space

    nn_network = NNModel(in_dim=observation_space.shape[0], out_dim=action_space.n)
    optimizer = torch.optim.Adam(nn_network.parameters(), lr=learning_rate)
    
    prediction_network = DQN(actions=action_space.n, network=nn_network, optimizer=optimizer, loss_function=dqn_loss).to(device)
    target_network = deepcopy(prediction_network)

    replay_buffer = ReplayBuffer(buffer_size)
    
    r_cumuls = np.zeros(epochs)
    loss_cumuls = np.zeros(epochs)

    for n_epoch in range(epochs):
        s = environment.reset()

        for t in range(T):
            a = prediction_network.get_action(s, epsilon)
            s_, r, is_terminal_state, _ = environment.step(a)
            
            r_cumuls[n_epoch] += r

            replay_buffer.store((s,a,r,s_,is_terminal_state))
            s = s_

            if (len(replay_buffer) < start_replay):
                continue

            if t % training_interval == 0:
                batch = replay_buffer.get_batch(batch_size)
                states, (chosen_actions, targets) = format_batch(batch, target_network, gamma)
                loss_cumuls[n_epoch] += prediction_network.train_on_batch(states, (chosen_actions, targets))

                target_network.soft_update(prediction_network, tau)

            if is_terminal_state or r_cumuls[n_epoch] > 200:
                print(f"Is terminal state? {is_terminal_state}\t n_epoch={n_epoch} step={t} r_cumul={r_cumuls[n_epoch]} loss_cumul={loss_cumuls[n_epoch]} epsilon={epsilon}")
                break
        
        epsilon = max(0.99 * epsilon, epsilon_min)
    
    environment.close()

    return r_cumuls, loss_cumuls

    
if __name__ == "__main__":
    '''
    All hyperparameter values and overall code structure are only given as a baseline. 
    
    You can use them if they help  you, but feel free to implement from scratch the
    required algorithms if you wish!
    '''
    import time
    from matplotlib import pyplot as plt

    batch_size = 64
    gamma = 0.99
    buffer_size = 1e6
    seed = 42
    tau = 0.01
    training_interval = 2
    learning_rate = 1e-4

    start = time.time()

    r_cumuls, loss_cumuls = run(batch_size, gamma, buffer_size, seed, tau, training_interval, learning_rate)

    end = time.time()
    print(f"Run duration = {(end-start)/60} min")
    
    plt.figure()
    plt.plot(r_cumuls, label="Cumulative return per epoch")
    plt.xlabel("n_epoch")
    plt.legend()
    
    plt.figure()
    plt.plot(loss_cumuls, label="Cumulative loss per epoch")
    plt.xlabel("n_epoch")
    plt.legend()
    plt.show()
