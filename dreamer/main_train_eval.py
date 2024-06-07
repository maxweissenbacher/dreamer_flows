#not working yet
import sys
import warnings
import dreamerv3
from dreamerv3 import embodied

def make_ks_env(config):
  import gym
  from embodied.envs import from_gym
  from gym.wrappers.time_limit import TimeLimit
  from ks.KS_environment import KSenv
  import numpy as np

  env = TimeLimit(KSenv(nu=0.08,
              actuator_locs=np.linspace(0.2, 2 * np.pi - 0.2, 7),#np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
              # sensor_locs=np.array([0, 2 * np.pi, 64]),
              burn_in=1500), max_episode_steps = config.KS.max_episode_steps)

  env = from_gym.FromGym(env, obs_key='vector')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)
  
  return env

def main(keyword_args):

  
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  
  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])

  config = config.update({
  'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/notebook_test_ks_2',
  'run.train_ratio': 64,
  'run.log_every': 30,  # seconds
  'batch_size': 16,
  'batch_length': 64,
  'jax.prealloc': False,
    
  # 'rssm.deter': 128,
  #   '.*\.cnn_depth': 32
  # '.*\.units': keyword_args["units"],
  # '.*\.layers': keyword_args["layers"],
    
  'encoder.mlp_keys': 'vector',
  'decoder.mlp_keys': 'vector',    
  'encoder.cnn_keys': '$^',
  'decoder.cnn_keys': '$^',
        
  'model_opt.lr': 1e-4,
    
  'jax.platform': 'cpu',
  'wrapper.length': 0,
  })

  config = embodied.Flags(config).parse()

  print("Number of Envs: ", config.envs.amount)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  

  replay = embodied.replay.Uniform(
                config.batch_length, config.replay_size, logdir / 'replay')
  # eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
  eval_replay = embodied.replay.Uniform(
                config.batch_length, config.replay_size, logdir / 'eval_replay')
  env = make_ks_env(config)
  eval_env = make_ks_envs(config)  # mode='eval'
  agent = agt.Agent(env.obs_space, env.act_space, step, config)
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train_eval(
      agent, env, eval_env, replay, eval_replay, logger, args)

  # agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  # replay = embodied.replay.Uniform(
  #     config.batch_length, config.replay_size, logdir / 'replay')
  # args = embodied.Config(
  #     **config.run, logdir=config.logdir,
  #     batch_steps=config.batch_size * config.batch_length)
  # embodied.run.train(agent, env, replay, logger, args)

  #eval_only
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  
  
  #for model size
  
  keyword_args = {}
  # --.*\.units=256 --.*\.layers=2
  # for arg in sys.argv[1:]:
  #     print(arg)
  #     if ".*." in arg:
  #         key, value = arg.split('=', 1)
  #         key = key[5:]
  #         keyword_args[key] = value
          
  main(keyword_args)
