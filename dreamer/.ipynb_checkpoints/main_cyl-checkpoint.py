def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  print("Reading config", flush =True)

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/cyl_test_run_wl0',
      'run.train_ratio': 64,
      'run.log_every': 30,  # seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': 'vector',
      'decoder.mlp_keys': 'vector',
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
      'jax.platform': 'cpu',
      'wrapper.length': 0,
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

#   import crafter
  import gym
  from embodied.envs import from_gym

  from cyl.simulation_base.env import resume_env
  import numpy as np

  print("Setting Up Environment", flush =True)
  env  = resume_env(plot=False, dump_CL=True, dump_vtu=500, dump_debug=10)

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
