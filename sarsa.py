import numpy as np
import gym
from tilecoding import TileCoder


def get_tile_coder(environment):
    return TileCoder(
        environment.observation_space.high,
        environment.observation_space.low,
        num_tilings=8,
        tiling_dim=8,
        max_size=4096,
    )


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)

def choose_action(state, action_space, theta, tile_coder):
    Q = []
    for action in range(action_space.n):
        x = tile_coder.phi(state, action)
        Q.append(np.sum(theta[x]))

    return np.argmax(Q)
    
def run(seed, epochs=100, T=200, alpha=0.1, lambda_=0.9, gamma=1.0):
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    tile_coder = get_tile_coder(environment)
    
    theta = np.zeros(tile_coder.size)
    
    r_cumuls = np.zeros(epochs)
    r_cumuls_means = []

    action_space = environment.action_space

    for n_epoch in range(epochs):
        s = environment.reset()
        a = choose_action(s, action_space, theta, tile_coder)
        # Traces
        z = np.zeros(tile_coder.size)

        for _ in range(T):
            s_, r, is_terminal_state, _ = environment.step(a)
            r_cumuls[n_epoch] += r
            delta = r

            for x in tile_coder.phi(s, a):
                delta -= theta[x]
                # Replacing traces
                z[x] = 1

            if is_terminal_state:
                # print(f"epoch {n_epoch} skipped with r_cumul = {r_cumuls[n_epoch]}")
                theta += alpha * delta * z
                break

            a_ = choose_action(s_, action_space, theta, tile_coder)

            for x in tile_coder.phi(s_, a_):
                delta += gamma * theta[x]
            
            theta += alpha * delta * z
            z *= gamma * lambda_
            
            s, a = s_, a_

        if n_epoch >= 99 and np.mean(r_cumuls[n_epoch-99:n_epoch+1]) >= -110:
            print(f"Environment resolved with r_cumuls mean = {np.mean(r_cumuls[n_epoch-99:n_epoch+1])}, n_epoch = {n_epoch}")
            break
        
        r_cumuls_means.append(np.mean(r_cumuls[n_epoch-99:n_epoch+1]))
    
    environment.close()

    return r_cumuls_means


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    seed = 42 
       
    for l in np.linspace(0,1,5):
        print(f"lambda = {str(l)}")
        plt.plot(run(seed, epochs=500, lambda_=l), label="{0} = {1}".format(r"$\lambda$", str(l)))

    plt.title("Cumulative reward mean over the 100 steps")
    plt.xlabel("n_epoch")
    plt.ylabel("Cumul rewards mean")
    plt.yticks(np.linspace(-200, -100, 11))    
    plt.legend()    
    plt.show()