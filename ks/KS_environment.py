# Here we wrap the numerical KS solver into a TorchRL environment

from typing import Optional
import numpy as np
from ks.KS_solver import KS
import gym
from gym import spaces


class KSenv(gym.Env):
    metadata = {}

    def __init__(
            self,
            nu,
            actuator_locs,
            sensor_locs,
            burn_in=0,
            target=None,
            frame_skip=1,
            soft_action=False,
            autoreg_weight=0.0,
            actuator_loss_weight=0.0,
            initial_amplitude=1e-2,
            actuator_scale=0.1,
            seed=None,
            device="cpu"):
        # Specify simulation parameters
        self.nu = nu
        self.N = 64
        self.dt = 0.05
        self.action_size = actuator_locs.shape[-1]
        self.actuator_locs = actuator_locs
        self.actuator_scale = actuator_scale
        self.burn_in = burn_in
        self.initial_amplitude = initial_amplitude
        self.observation_inds = [int(x) for x in (self.N / (2 * np.pi)) * sensor_locs]
        self.num_observations = len(self.observation_inds)
        assert len(self.observation_inds) == len(set(self.observation_inds))
        self.termination_threshold = 20.  # Terminate the simulation if max(u) exceeds this threshold
        self.action_low = -1.0  # Minimum allowed actuation (per actuator)
        self.action_high = 1.0  # Maximum allowed actuation (per actuator)
        self.actuator_loss_weight = actuator_loss_weight
        self.soft_action = soft_action
        self.autoreg_weight = autoreg_weight
        self.frame_skip = frame_skip
        self.device = device
        if target is None:  # steer towards zero solution
            self.target = np.zeros(self.N)
        elif target == 'u1':
            self.target = np.tensor(np.loadtxt('../../../solver/solutions/u1.dat'), device=self.device)

        super().__init__()

        self.solver_step = KS(nu=self.nu,
                              N=self.N,
                              dt=self.dt,
                              actuator_locs=self.actuator_locs,
                              actuator_scale=self.actuator_scale,
                              device=self.device,
                              ).advance

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(self.action_size,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))

    def step(self, action):
        u = self.u  # Solution at previous timestep
        reward_sum = np.zeros([])
        for i in range(self.frame_skip):  # Take frame_skip many steps
            u = self.solver_step(u, action)  # Take a step using the PDE solver
            # reward = - (L2 norm of solution + hyperparameter * L2 norm of action)
            reward = - np.linalg.norm(u-self.target, axis=-1) - self.actuator_loss_weight * np.linalg.norm(action, axis=-1)
            reward_sum += reward
        reward = reward_sum / self.frame_skip  # Compute the average reward over frame_skip steps
        #reward_mean = reward_mean.view(*tensordict.shape, 1)
        self.u = u
        observation = u[self.observation_inds]  # Evaluate at desired indices
        # To allow for batched computations, use this instead:
        # ... however the KS solver needs to be compatible with np.vmap!
        # u = np.vmap(self.solver_step)(u, action)
        done = np.max(np.abs(u)) > self.termination_threshold
        #done = done.view(*tensordict.shape, 1)

        return observation, reward, done, {}

    def reset(self):
        # Initial data drawn from IID normal distributions
        zrs = np.zeros([self.N])
        ons = np.ones([self.N])
        u = np.random.normal(zrs, ons)
        u = self.initial_amplitude * u
        u = u - u.mean(axis=-1)
        # Burn in
        for _ in range(self.burn_in):
            u = self.solver_step(u, np.zeros(self.action_size, device=self.device))
        # Store the solution in class variable
        self.u = u
        # Compute observation
        observation = u[self.observation_inds]
        return observation


if __name__ == '__main__':

    # Defining env
    env = KSenv(nu=0.08,
                actuator_locs=np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
                sensor_locs=np.array([0.0, 1.0, 2.0]),
                burn_in=0)
    env.reset()
    print('hi')
    env.step(action=np.zeros(4))


