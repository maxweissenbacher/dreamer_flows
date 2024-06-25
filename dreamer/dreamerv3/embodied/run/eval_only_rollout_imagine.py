import re
import tensorflow as tf
import embodied
import numpy as np


def eval_only_rollout_imagine(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  # logdir.mkdirs()
  print('Logdir', logdir)
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  # timer = embodied.Timer()
  # timer.wrap('agent', agent, ['policy'])
  # timer.wrap('env', env, ['step'])
  # timer.wrap('logger', logger, ['write'])
  
  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key]# The `max` function in the provided code snippet is used to
        # find the maximum value along a specified axis.
        max(0).mean()
    # metrics.add(stats, prefix='stats')

  
  def eval_rollout_log(tran):
      # print(tran.keys())
      # logger.add({'action': tran["action"], 'state': tran["vector"]}, prefix='rollout_eval_episode')
      metrics.add({'reward': tran['reward']}, prefix='rollout_eval_episode')  
  
  driver = embodied.Driver(env)
  # driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: eval_rollout_log(tran))
  driver.on_step(lambda tran, _: step.increment())

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset()
  #unrolling
  eval_eps_index = 0
  while step < args.steps:
    # driver(policy, steps=100, eval = True)
    driver(policy_eval, steps = args.eval_steps)
    eval_eps_index+=1
  #calculating and logging mean reward for eval episode
  eval_eps_reward = np.array(metrics.get_key("rollout_eval_episode/reward"))
  mean_eval_eps_reward = np.mean(eval_eps_reward)
  last_eval_eps_reward = eval_eps_reward[-1]
  logger.add({'mean_reward': mean_eval_eps_reward, 'last_reward': last_eval_eps_reward}, prefix='rollout_eval_episode')
  logger.write()

        
      