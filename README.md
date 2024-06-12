# Installation

Run 'pip install -r requirements.txt'

# Running

The main is in dreamer/main.py

# KS Parameters
nu=0.08,
actuator_locs=np.linspace(0.2, 2 * np.pi - 0.2, 20)
actuator_scale=0.1,
frame_skip=1,
burn_in=2000,
max_episode_steps = 500

# Note
move to gym==0.25.0 for TimeLimit wrapper to work
