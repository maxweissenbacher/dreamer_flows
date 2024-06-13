#not working yet
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def main(keyword_args):

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  
  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  # config = config.update(dreamerv3.configs['KS'])

  config = config.update({
  'logdir': '~/PhD_projects2/dreamer_flows/dreamer/logdir/ks_test_run_negreward_wl100_eval',
  'run.train_ratio': 64,
  'run.log_every': 30,  # seconds
  'batch_size': 16,
  'batch_length': 64,
  'jax.prealloc': False,
    
  # 'rssm.deter': 128,
  # #   '.*\.cnn_depth': 32
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
  logdir_name = config.logdir_basepath+'/'+\
           config.logdir_dirname+'/'+\
           config.logdir_expname
  config = config.update({'logdir': logdir_name})
  
  logdir = embodied.Path(config.logdir)
  logdir.mkdirs()
  config.save(config.logdir+"/config.yaml")
  print('Logdir', logdir)
  print("Number of Envs: ", config.envs.amount)

  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

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

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  
  embodied.run.eval_only_rollout(agent, env, logger, args)

  #eval_only
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  
  
  #for model size
  
  keyword_args = {}
  # # --.*\.units=256 --.*\.layers=2
  # for arg in sys.argv[1:]:
  #   #   print(arg)
  #     if ".*." in arg:
  #         key, value = arg.split('=', 1)
  #         key = key[5:]
  #         keyword_args[key] = value
          
  main(keyword_args)
