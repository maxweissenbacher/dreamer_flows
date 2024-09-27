#not working yet
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import warnings
import dreamerv3
from dreamerv3 import embodied
import wandb

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
  'encoder.mlp_keys': 'vector',
  'decoder.mlp_keys': 'vector',    
  'encoder.cnn_keys': '$^',
  'decoder.cnn_keys': '$^',   
  'model_opt.lr': 1e-4,
  'wrapper.length': 0,
  })

  config = embodied.Flags(config).parse()
  config = config.update({'grad_heads': gradcontrol(config)})
  print("############# config gradhead: ", config.grad_heads)
  print("############  "+ config.logdir_eval+"  ###################")
  logdir_name = config.logdir_basepath+'/'+\
                config.logdir_dirname+'/'+\
                config.logdir_expname  
  config = config.update({'logdir': logdir_name, 
                          'logdir_eval': logdir_name+'_Eval'+f"nu{config.KS.nu}_"+config.logdir_info,
                          'run.from_checkpoint': logdir_name + "/checkpoint.ckpt"
                          })
  logdir = embodied.Path(config.logdir)
  logdir_eval = embodied.Path(config.logdir_eval)
  logdir_eval.mkdirs()
  
  # import os.path
  # if os.path.isfile(config.logdir+"/config.yaml"):
     
  
  config.save(config.logdir_eval+"/config.yaml")
  print("##########################################")
  print('LOGDIR', config.logdir)
  print("Number of Envs: ", config.envs.amount)
  print("##########################################")
  
  # ############################ Creating logger ##############################
  
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir_eval, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir_eval),
      embodied.logger.WandBOutput(
            pattern="$",
            logdir=logdir_eval,
            config=config,
        ),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])
  
  
  #make replay
#   train_replay = embodied.replay.Uniform(
#                 config.batch_length, config.replay_size, logdir / 'eval_replay')
  eval_replay = embodied.replay.Uniform(
                config.batch_length, config.replay_size, logdir / 'eval_replay')
  
  #make env
  from make_flow_envs import make_flow_envs, make_ks_env
  eval_env = make_flow_envs(config, env_name="KS", num_envs = config.run.num_eval_envs)

  agent = dreamerv3.Agent(eval_env.obs_space, eval_env.act_space, step, config)
  args = embodied.Config(
      **config.run, logdir=config.logdir, logdir_eval = config.logdir_eval,
      batch_steps=config.batch_size * config.batch_length)

    
  ########################### Run Training or eval ##############################
  embodied.run.eval_rollout(
          agent, eval_env, eval_replay, logger, args)


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
  
  
  keyword_args = parse_model_size()   
  main(keyword_args)
