def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  # config = embodied.Config(dreamerv3.configs['defaults'])
  # config = config.update(dreamerv3.configs['medium'])
  # config = config.update({
  #     'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/ks_test_run_negreward_wl0_new',
  #     'run.train_ratio': 32,
  #     'run.log_every': 30,  # seconds
  #     'batch_size': 16,
  #     'batch_length': 30,
  #     'jax.prealloc': False,
  #     'encoder.mlp_keys': 'vector',
  #     'decoder.mlp_keys': 'vector',
  #     'encoder.cnn_keys': '$^',
  #     'decoder.cnn_keys': '$^',
  #     'jax.platform': 'cpu',
  #     'wrapper.length': 0,
  # })
  
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/ks_test_run_negreward_wl0_smalltrainratio_medium',
      'run.train_ratio': 2,
      'run.log_every': 30,  # seconds
      'batch_size': 16,
      'batch_length': 6,
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

  from ks.KS_environment import KSenv
  import numpy as np

  env = KSenv(nu=0.08,
              actuator_locs=np.linspace(0.2, 2 * np.pi - 0.2, 7),#np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
              # sensor_locs=np.array([0, 2 * np.pi, 64]),
              burn_in=100)

  # env = gym.make('CartPole-v1')
  # env = crafter.Env()  # Replace this with your Gym env.


  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  
  #eval_only
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  
  main()
