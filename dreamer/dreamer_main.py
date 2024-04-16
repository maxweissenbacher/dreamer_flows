def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  warnings.warn('Make sure the correct environment is activated. See requirements.txt.', UserWarning)

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/test_run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': 'vector',
      'decoder.mlp_keys': 'vector',
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
      'jax.platform': 'cpu',
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

  import crafter
  import gym
  from embodied.envs import from_gym

  env = gym.make('CartPole-v1')
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
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
