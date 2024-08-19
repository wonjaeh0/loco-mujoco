from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import get_gradient, zero_grad, to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset, compute_J, arrays_as_dataset, compute_episodes_length
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO
from mushroom_rl.utils.minibatches import minibatch_generator

from imitation_lib.utils import GailDiscriminatorLoss, to_float_tensors


class PPOagent(PPO):
    """
    PPO agent

    """

    def __init__(self, mdp_info, policy_class, policy_params, sw,
                 actor_optimizer, critic_params, n_epochs_policy, batch_size, eps_ppo, lam=0.97):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(PPOagent, self).__init__(mdp_info=mdp_info, policy=policy, actor_optimizer=actor_optimizer, critic_params=critic_params,
                                n_epochs_policy=n_epochs_policy, batch_size=batch_size, eps_ppo=eps_ppo, lam=lam, ent_coeff=0.0,
                                critic_fit_params=None)

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._add_save_attr(
            _discriminator_fit_params='pickle',
            _loss='torch',
            _train_n_th_epoch ='pickle',
            _D='mushroom',
            _env_reward_frac='primitive',
            _demonstrations='pickle!',
            _act_mask='pickle',
            _state_mask='pickle',
            _use_next_state='pickle',
            _use_noisy_targets='pickle',
            _trpo_standardizer='pickle',
            _D_standardizer='pickle',
            _train_D_n_th_epoch='pickle',
            _n_epochs_discriminator="primitive",
            ext_normalizer='pickle',
        )

        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer,step_size=2000,gamma=0.5)

    def fit(self, dataset, **info):
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)
        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda())
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        old_pol_dist = self.policy.distribution_t(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        self._V.fit(x, v_target, **self._critic_fit_params)

        self._update_policy(obs, act, adv, old_log_p)
        self._scheduler.step()

        # Print fit information
        # self._log_info(dataset, x, v_target, old_pol_dist)
        self._logging_sw(dataset, x, v_target, old_pol_dist)
        self._iter += 1
            
    def _logging_sw(self, dataset, x, v_target, old_pol_dist):
        if self._sw:
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            v_pred = torch.tensor(self._V(x), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)

            logging_ent = self.policy.entropy(x)
            new_pol_dist = self.policy.distribution(x)
            logging_kl = torch.mean(
                torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist)
            )
            avg_rwd = np.mean(compute_J(dataset))
            L = int(np.round(np.mean(compute_episodes_length(dataset))))

            self._sw.add_scalar('EpTrueRewMean', avg_rwd, self._iter)
            self._sw.add_scalar('EpLenMean', L, self._iter)
            self._sw.add_scalar('vf_loss', v_err, self._iter)
            self._sw.add_scalar('entropy', logging_ent, self._iter)
            self._sw.add_scalar('kl', logging_kl, self._iter)

    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i, act_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_i)
                loss.backward()
                self._optimizer.step()

    def _post_load(self):
        self._sw = None