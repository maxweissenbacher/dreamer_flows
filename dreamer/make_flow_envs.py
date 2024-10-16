import importlib
import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

# directory = pathlib.Path(__file__).resolve()
# directory = directory.parent
# sys.path.append(str(directory.parent))
# sys.path.append(str(directory.parent.parent))
# sys.path.append(str(directory.parent.parent.parent))
# __package__ = directory.name

import dreamerv3
from dreamerv3 import embodied
from embodied import wrappers
from embodied.envs import from_gym
import numpy as np

def make_ks_env(config, n_env = 0, sim_log_name = "KS", mode = None):
  import gym
  from gym.wrappers.time_limit import TimeLimit
  from ks.KS_environment import KSenv
  
  env = TimeLimit(KSenv(nu=config.KS.nu,
              actuator_locs=np.linspace(0.2, 2 * np.pi - 0.2, config.KS.num_actuators),#np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
              actuator_scale=config.KS.actuator_scale,
              frame_skip=config.KS.frame_skip,
              # sensor_locs=np.array([0, 2 * np.pi, 64]),
              burn_in=config.KS.burn_in), max_episode_steps = config.KS.max_episode_steps)
  
  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  return env

def make_cyl_env(config, n_env = 0, sim_log_name = "Test_cylinder", mode='train'):
  import gym
  from gym.wrappers.time_limit import TimeLimit
  from Cylinder_Env.simulation_base.env import resume_env
  import numpy as np
  
  sim_log_name = config.logdir.split("/")[-1]
  # env  = resume_env(plot=False, dump_CL=False, \
  #                   dump_vtu=False, dump_debug=10, \
  #                   sim_log_name = sim_log_name)
  
  if mode == "train":
    env = resume_env(plot=False,
                      single_run=False,
                      horizon=config.cyl.horizon,
                      dump_vtu=config.cyl.dump_vtu,
                      dump_debug = config.cyl.dump_debug, 
                      random_start= config.cyl.random_start,
                      n_env=n_env,
                      simulation_duration=config.cyl.simulation_duration,
                      sim_log_name = sim_log_name+"/train"
                      )
  elif mode == "eval":
    env = resume_env(plot=False,
                      single_run=False,
                      horizon=config.cyl.horizon,
                      dump_vtu=config.cyl.eval_dump_vtu,
                      dump_debug = config.cyl.eval_dump_debug, 
                      random_start= config.cyl.random_start,
                      n_env=n_env,
                      simulation_duration=config.cyl.simulation_duration,
                      sim_log_name = sim_log_name+"/eval"
                      )
  
  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  
  return env

 
def make_flow_envs(config, env_name = "KS", num_envs = 1, mode = None):
  
  assert env_name == "KS" or env_name == "CYL",\
          "Env name should be KS or CYL"
          
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(num_envs):
    print(f"in loop env {index}")
    make_env = make_cyl_env if env_name == "CYL" else make_ks_env
    ctor = lambda index=index: make_env(config, n_env = index, 
                            sim_log_name= config.logdir_dirname+"/"+config.logdir_expname,
                            mode = mode)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


if __name__ == '__main__':
  
  print("Creates envs for KS and CYLINDER")



# def make_cylinder_env(device, cfg, n_env=1, sim_log_name = "Sim"):
#     from torchrl.envs.libs.gym import GymWrapper
#     # Create the 2D cylinder Gym environment here
#     env = resume_env(plot=False,
#                      single_run=False,
#                      horizon=cfg.cyl.horizon,
#                      dump_vtu=cfg.cyl.dump_vtu,
#                      dump_debug = cfg.cyl.dump_debug, 
#                      random_start= cfg.cyl.random_start,
#                      n_env=n_env,
#                      simulation_duration=cfg.cyl.simulation_duration,
#                      sim_log_name = sim_log_name
#                      )
#     env = GymWrapper(env, device=device)
#     return env

# def make_parallel_cylinder_env(cfg, exp_name):
#     # make_env_fn = EnvCreator(lambda: make_cylinder_env(cfg.collector.device, cfg))
#     # env = ParallelEnv(cfg.cyl.num_envs, make_env_fn)
    
#     # env = ParallelEnv(cfg.cyl.num_envs, make_env_fn)
#     env = ParallelEnv(cfg.env.num_envs, 
#                       [lambda: make_cylinder_env(cfg.collector.device, cfg, n_env, sim_log_name = exp_name + "/train") for n_env in range(cfg.env.num_envs)])  
#     # env = ParallelEnv(cfg.cyl.num_envs, [lambda: make_ks for i in range(num_envs)])
#     return env


# def make_cylinder_eval_env(cfg, exp_name):
#     test_env = make_cylinder_env(cfg.collector.device, cfg, sim_log_name = exp_name + "/eval")
#     test_env.eval()
#     return test_env