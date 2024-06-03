# Conservative Q-Learning for Offline Reinforcement Learning 2020 [CQL]
# https://arxiv.org/abs/2006.04779
# The code is from https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch import autograd
from armin_utils.pytorch import tensor
torch.autograd.set_detect_anomaly(True)


class CQL:
    def __init__(
            self, 
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            action_dim,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0):
        
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.action_dim = action_dim
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.action_dim).item()
            self.log_alpha = tensor.zeros(1, grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)
        
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = tensor.zeros(1, grad=True)
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)


        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1
        self.num_qs = num_qs
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.discrete = False
    
    
    def train2(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)
        
        
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds


    def _get_policy_actions(self, obs, num_actions, network=None):
        print('obs.size()', obs.size())
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        print('obs_temp.size()', obs_temp.size())
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(obs_temp, reparameterize=True, return_log_prob=True)
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions


    def train(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        truncations = batch['truncations']
        terminals = batch['truncations']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        actions_policy, policy_mean, policy_log_std, log_pi, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, actions_policy)
        else:
            q_new_actions = torch.min(self.qf1(obs, actions_policy), self.qf2(obs, actions_policy))

        policy_loss = (alpha*log_pi - q_new_actions).mean()
        # print('Policy Loss', policy_loss)
        if self._current_epoch < self.policy_eval_start:
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (alpha*log_pi - policy_log_prob).mean()
            

        q1_pred = self.qf1(obs, actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(obs, actions)
        
        new_next_actions, _, _, new_log_pi, *_ = self.policy(next_obs, reparameterize=True, return_log_prob=True)
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)

        if not self.max_q_backup:
            if self.num_qs == 1:
                target_q_values = self.target_qf1(next_obs, new_next_actions)
            else:
                target_q_values = torch.min(self.target_qf1(next_obs, new_next_actions), self.target_qf2(next_obs, new_next_actions))
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
            
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target)

        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        cat_q1 = torch.cat([q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
        cat_q2 = torch.cat([q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat([q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1)
            cat_q2 = torch.cat([q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1)
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        
        if self.num_qs > 1:
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            policy_loss = (log_pi - q_new_actions).mean()

        self._n_train_steps_total += 1


    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True


    @property
    def networks(self):
        base_list = [self.policy,self.qf1,self.qf2,self.target_qf1,self.target_qf2]
        return base_list


    def get_snapshot(self):
        return dict(policy=self.policy,qf1=self.qf1,qf2=self.qf2,target_qf1=self.target_qf1,target_qf2=self.target_qf2)