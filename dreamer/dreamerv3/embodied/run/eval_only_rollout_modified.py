import re
import embodied
import numpy as np
import copy

def eval_rollout(
    agent, eval_env, eval_replay, logger, args):

  logdir_eval = embodied.Path(args.logdir_eval)
  logdir = embodied.Path(args.logdir)
  
  should_expl = embodied.when.Until(args.expl_until)
  # should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log  = embodied.when.Every(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  
  # should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step    = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  # eval_metrics = embodied.Metrics()
  print('Observation space:', embodied.format(eval_env.obs_space), sep='\n')
  print('Action space:', embodied.format(eval_env.act_space), sep='\n')

  # timer = embodied.Timer()
  # timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  # timer.wrap('env', train_env, ['step'])
  # if hasattr(train_replay, '_sample'):
  #   timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length, 
        'score': score, 
        'sum_abs_reward': sum_abs_reward,
        'reward': ep['reward'],
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
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
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  def eval_rollout_log(tran, worker):
    if not hasattr(eval_rollout_log, "counter"):
      eval_rollout_log.counter = 0
      eval_rollout_log.run_num = 0

    if eval_rollout_log.counter == args.eval_rollout_steps*args.num_eval_envs:
      eval_rollout_log.run_num +=1
      eval_rollout_log.counter = 0
    eval_rollout_log.counter += 1
    
    run_num = eval_rollout_log.run_num
    # print(tran.keys())
    logger.add({f'rollout_reward_{worker+(run_num*args.num_eval_envs)}': tran["reward"]}, prefix='rollout_eval_episode')
    metrics.add({f'reward_{worker}': tran['reward']}, prefix='rollout_eval_episode')
  
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_step(lambda tran, worker: eval_rollout_log(tran, worker))
  driver_eval.on_step(lambda tran, _: step.increment())
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))
  
  
  random_agent = embodied.RandomAgent(eval_env.act_space)

  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()
  
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  
  step_old = copy.deepcopy(step)
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = eval_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)

  # print("############ "+ args.KS.nu + " ################")
  print('Start evaluation loop.')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  # while step < args.steps:
  #   if should_eval(step):
  #     print('Starting evaluation at step', int(step))
  run_num = 0
  while run_num < (args.num_eval_rollouts // args.num_eval_envs):
    #unrolling
    step.load(0)
    driver_eval.reset()
    eval_eps_index = 0
    while eval_eps_index < args.eval_eps:
      driver_eval(policy_eval, steps = args.eval_rollout_steps*args.num_eval_envs)
      eval_eps_index+=1
    run_num += 1

  eval_eps_rewards = np.array([metrics.get_key(k) \
                                for k in metrics.get_metric_keys() \
                                if "rollout_eval_episode/reward" in k])
  print("eval_eps_rewards shape: ", eval_eps_rewards.shape)
  print(metrics.get_metric_keys())
  mean_eval_eps_rewards = np.mean(eval_eps_rewards)
  std_eval_eps_rewards  = np.std(eval_eps_rewards)
  last_eval_eps_rewards = np.mean(eval_eps_rewards[:,-1])
  std_last_eval_eps_rewards  = np.std(eval_eps_rewards[:,-1])

  logger.add({'mean_rewards'      : mean_eval_eps_rewards,
              'std_rewards'       : std_eval_eps_rewards,
              'mean_last_rewards' : last_eval_eps_rewards, 
              'std_last_rewards'  : std_last_eval_eps_rewards}, prefix='rollout_eval_episode')
  
  for i in range(eval_eps_rewards.shape[0]):
      logger.add({f'last_reward_rollout{i}': eval_eps_rewards[i,-1]}, prefix='rollout_eval_episode')
      logger.add({f'mean_reward_rollout{i}': np.mean(eval_eps_rewards[i,:])}, prefix='rollout_eval_episode')
    
  logger.write()
  
  
