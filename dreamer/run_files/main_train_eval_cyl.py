#working 
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import warnings
import dreamerv3
from dreamerv3 import embodied
import wandb

import os


def main(keyword_args):

  ############################Load Config##############################
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['large'])
  config = config.update({
  # 'rssm.deter': 128,
  # #   '.*\.cnn_depth': 32
  # '.*\.units': keyword_args["units"],
  # '.*\.layers': keyword_args["layers"],
  'run.steps': 2e5,
  'encoder.mlp_keys': 'vector',
  'decoder.mlp_keys': 'vector',    
  'encoder.cnn_keys': '$^',
  'decoder.cnn_keys': '$^',   
  'model_opt.lr': 1e-5,
  'wrapper.length': 0,
  })

  config = embodied.Flags(config).parse()
  config = config.update({'grad_heads': gradcontrol(config)})
  print("############# config gradhead: ", config.grad_heads)
  
  #only for generalizability test
#   config = config.update({'actor.layers': 2, 'actor.units': 32,
#                           'critic.layers': 2, 'critic.units': 32,
#                           'encoder.mlp_layers': 3, 'encoder.mlp_units': 512,
#                           'decoder.mlp_layers': 3, 'decoder.mlp_units': 512,
#                           'rssm.deter': 1024, 'rssm.units': 512
#                           }
#                          )
  
  logdir_name = config.logdir_basepath+'/'+\
                config.logdir_dirname+'/'+\
                config.logdir_expname  
                
  config = config.update({'logdir': logdir_name})
  logdir = embodied.Path(config.logdir)
  logdir.mkdirs()
  
  # import os.path
  # if os.path.isfile(config.logdir+"/config.yaml"):
     
  config.save(config.logdir+"/config.yaml")
  print("##########################################")
  print('LOGDIR', config.logdir)
  print("Number of Envs: ", config.envs.amount)
  print("##########################################")
  
#   wandb.init(
#         project=config.wandb.project,
#         name=config.logdir,
#         # sync_tensorboard=True,,
#         entity='why_are_all_the_good_names_taken_aaa',
#         config=dict(config),
#     )

  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(
            pattern="$",
            logdir=logdir,
            config=config,
        )
    #   embodied.logger.WandBOutput(
    #         pattern="$",
    #         logdir=logdir,
    #         project=config.wandb.project,
    #         entity=config.wandb.entity,
    #         # mode = "offline",
    #         config=config
    #     )
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  
  ############################ Creating Env ##############################
  
  
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

def gradcontrol(config):
    grad_heads = ['decoder']
    if config.use_rewardmodel:
        grad_heads.append('reward')
    if config.use_cont:
        grad_heads.append('cont')
    return grad_heads
  
if __name__ == '__main__':

#   os.environ['LD_LIBRARY_PATH'] = '~/anaconda3/envs/dreamer_cyl2/lib/:$LD_LIBRARY_PATH'
#  os.system("echo $LD_LIBRARY_PATH")
  keyword_args = parse_model_size()
  main(keyword_args)

