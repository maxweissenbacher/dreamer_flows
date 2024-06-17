import re
import embodied
import numpy as np


def train_eval_rollout(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Every(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  # eval_metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  # timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

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
    # print(tran.keys())
    # logger.add({'action': tran["action"], 'state': tran["vector"]}, prefix='rollout_eval_episode')
    metrics.add({f'reward_{worker}': tran['reward']}, prefix='rollout_eval_episode')
  
  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_step(lambda tran, worker: eval_rollout_log(tran, worker))
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
      
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      
      # driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
      # eval_eps_rewards = np.empty((0,args.eval_steps))
      # print(eval_eps_rewards.shape)
      for run_num in range(args.num_eval_rollouts // args.num_eval_envs):
        #unrolling
        driver_eval.reset()
        eval_eps_index = 0
        while eval_eps_index < args.eval_eps:
          driver_eval(policy_eval, steps = args.eval_rollout_steps*args.num_eval_envs)
          eval_eps_index+=1
          
        # print(np.array(metrics.get_key("rollout_eval_episode/reward_0")).shape)
        # print(np.array(metrics.get_key("rollout_eval_episode/reward_1")).shape)
        # eval_eps_reward = np.array([metrics.get_key(k) \
        #                           for k in metrics.get_metric_keys() \
        #                           if "rollout_eval_episode/reward" in k])
        
        # print("eval_eps_reward: ", eval_eps_reward.shape)
        # eval_eps_rewards = np.concatenate((eval_eps_rewards, eval_eps_reward),axis=0)
      
      #calculating and logging mean reward for eval episode "rollout_eval_episode/reward"
      eval_eps_rewards = np.array([metrics.get_key(k) \
                                    for k in metrics.get_metric_keys() \
                                    if "rollout_eval_episode/reward" in k])
      print("eval_eps_rewards shape: ", eval_eps_rewards.shape)
      print(metrics.get_metric_keys())
      mean_eval_eps_rewards = np.mean(eval_eps_rewards)
      last_eval_eps_rewards = np.mean(eval_eps_rewards[:,-1])
      std_eval_eps_rewards  = np.std(eval_eps_rewards[:,-1])
      
      logger.add({'mean_rewards'      : mean_eval_eps_rewards, 
                  'mean_last_rewards' : last_eval_eps_rewards, 
                  'std_last_rewards'  : std_eval_eps_rewards}, prefix='rollout_eval_episode')

      # eval_eps_reward = np.array(metrics.get_key("rollout_eval_episode/reward"))
      # mean_eval_eps_reward = np.mean(eval_eps_reward)
      # last_eval_eps_reward = eval_eps_reward[-1]
      # logger.add({'mean_reward': mean_eval_eps_reward, 'last_reward': last_eval_eps_reward}, prefix='rollout_eval_episode')

    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
  
 
