#not working yet
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import warnings
import dreamerv3
from dreamerv3 import embodied

def main(keyword_args):

  ############################Load Config##############################
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  # config = config.update(dreamerv3.configs['multicpu'])

  config = config.update({
  'run.eval_every': 4000, 
  'run.train_ratio': 32,#64,
  'run.log_every': 30,  # seconds
  'batch_size': 8,#16,
  'batch_length': 8,#64,
  'jax.prealloc': False,
    
  # 'rssm.deter': 128,
  # #   '.*\.cnn_depth': 32
  '.*\.units': keyword_args["units"],
  '.*\.layers': keyword_args["layers"],
    
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

  ############################ Creating Env ##############################
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  
  #make replay
  replay = embodied.replay.Uniform(
                config.batch_length, config.replay_size, logdir / 'replay')
  # eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
  # eval_replay = embodied.replay.Uniform(
  #               config.batch_length, config.replay_size, logdir / 'eval_replay')
  #make env
  # env = make_ks_env(config)
  from make_flow_envs import make_flow_envs, make_cyl_env
  env = make_flow_envs(config, env_name="CYL", num_envs = config.envs.amount)
  eval_env = make_cyl_env(config, n_env=0, 
                          sim_log_name = config.logdir_dirname+"/"+\
                                         config.logdir_expname, mode = "eval")  # mode='eval'
  eval_env = embodied.BatchEnv([eval_env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  args  = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)

  ########################### Run Training or eval ##############################
  embodied.run.train_eval_rollout_noevalreplay(
          agent, env, eval_env, replay, logger, args)


def parse_model_size():
    #for parsing model size from terminal
    keyword_args = {}
    for arg in sys.argv[1:]:
        #   print(arg)
        if ".*." in arg:
            key, value = arg.split('=', 1)
            key = key[5:]
            keyword_args[key] = value
    return keyword_args

if __name__ == '__main__':
  
#   os.environ['LD_LIBRARY_PATH'] = '~/anaconda3/envs/dreamer_cyl2/lib/:$LD_LIBRARY_PATH'
  os.system("echo $LD_LIBRARY_PATH")
  keyword_args = parse_model_size()
  main(keyword_args)
