def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  print("Reading config", flush =True)

  # See configs.yaml for all options.
  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  print("Reading config", flush =True)

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
    # 'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/notebook_test_cyl_ks_8',
    'run.train_ratio': 32,
    'run.log_every': 30,  # seconds
    'batch_size': 16,
    'batch_length': 8,
    'jax.prealloc': False,
      
    'encoder.mlp_keys': 'vector',
  #   'encoder.mlp_units': 512,

    'decoder.mlp_keys': 'vector',
  #   'decoder.mlp_units': 512,
      
    'encoder.cnn_keys': '$^',
    'decoder.cnn_keys': '$^',
      
  #   'reward_head.units': 512,
  #   'cont_head.units': 512,
      
    'model_opt.lr': 1e-4,
      
    # 'jax.platform': 'cpu',
    'wrapper.length': 0,
      
    # 'envs.amount': 8
  })

  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    embodied.logger.TensorBoardOutput(logdir),

  ])

  print("confid jax.platform: ", config.jax.platform)

  #   import crafter
  import gym
  from embodied.envs import from_gym

  from cyl.simulation_base.env import resume_env
  import numpy as np

  print("Setting Up Environment", flush =True)
  sim_log_name = config.logdir.split("/")[-1]
  env  = resume_env(plot=False, dump_CL=False, dump_vtu=False, dump_debug=10, sim_log_name = sim_log_name)

  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  
  print("wrapping env", flush =True)
  env = dreamerv3.wrap_env(env, config)
  
  print("batching env", flush =True)
  env = embodied.BatchEnv([env], parallel=False)
  print("environment has been set", flush =True)
  
  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  print("agent has been created", flush =True)
  
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  print("replay has been created", flush =True)
  
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  print("args have been created", flush =True)
  
  print("Running training", flush =True)
  embodied.run.train(agent, env, replay, logger, args)
# embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  import sys
  import os
  cwd = os.getcwd()
  print(cwd)
  
  main()
