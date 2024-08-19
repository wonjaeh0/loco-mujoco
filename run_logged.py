from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv


env = LocoEnv.make("HumanoidTorque.run.real")

# agent = Agent.load("logs/loco_mujoco_evalution_2024-07-18_03-48-16/env_id___HumanoidMuscle.run.real/1/agent_epoch_91_J_107.866960.msh") # 1.9618148, 496
# agent = Agent.load("logs/loco_mujoco_evalution_2024-07-18_03-48-16/env_id___HumanoidMuscle.walk.real/1/agent_epoch_100_J_209.254482.msh") # 2.1538692, 1064
# agent = Agent.load("logs/loco_mujoco_evalution_2024-07-18_03-48-16/env_id___HumanoidMuscleExo.run.real/0/agent_epoch_96_J_107.271727.msh") # 1.9640547
# agent = Agent.load("logs/loco_mujoco_evalution_2024-07-18_03-48-16/env_id___HumanoidMuscleExo.walk.real/0/agent_epoch_125_J_210.020668.msh") # 2.2264295
agent = Agent.load("/home/wonjae/repo/loco-mujoco/logs/loco_mujoco_evalution_2024-08-18_16-03-56/env_id___HumanoidTorque.run.real/0/agent_epoch_73_J_127.608457.msh")
# agent = Agent.load("/home/wonjae/repo/loco-mujoco/logs/loco_mujoco_evalution_2024-08-04_09-53-08/env_id___HumanoidMuscle.run.real/1/agent_epoch_377_J_48.230931.msh")

core = Core(agent, env)

dataset, info = core.evaluate(n_episodes=30, render=True, get_env_info=True)

import numpy as np


obs = []
action = []
for a in dataset:
    # obs.append(a[0])
    # print(obs[-1][-3:])
    action.append(a[1])

import numpy as np
action = np.stack(action)
print(np.std(action))

import pdb
pdb.set_trace()


