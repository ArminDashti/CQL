# From https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py

import torch
import torch.nn as nn
from armin_utils.RL.env import minari
from armin_utils.RL.modules.policies.tanh_gaussian import tanh_gaussian
from armin_utils.utils.flatten_MLP import flatten_MLP
from CQL.model import CQL


env = minari.make()
rm = env.reply_memory()
obs_dim = env.obs_dim
action_dim = env.action_dim
t = rm.sample(3, tensor=True)

M = 256
policy = tanh_gaussian(obs_dim=obs_dim[0], action_dim=action_dim[0], hidden_sizes=[M, M, M]).float()
policy.train()
qf1 = flatten_MLP(input_size=obs_dim[0] + action_dim[0], output_size=1, hidden_sizes=[M, M, M]).float()
qf1.train()
qf2 = flatten_MLP(input_size=obs_dim[0] + action_dim[0], output_size=1, hidden_sizes=[M, M, M]).float()
qf2.train()
target_qf1 = flatten_MLP(input_size=obs_dim[0] + action_dim[0], output_size=1, hidden_sizes=[M, M, M]).float()
target_qf2 = flatten_MLP(input_size=obs_dim[0] + action_dim[0], output_size=1, hidden_sizes=[M, M, M]).float()

cql = CQL(policy=policy, qf1=qf1, qf2=qf2, target_qf1=target_qf1, target_qf2=target_qf2, action_dim=action_dim)

cql.train(rm.sample(64, tensor=True))
#%%
tt = rm.sample(64, tensor=True)