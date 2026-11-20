from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # (student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        with torch.no_grad():
            z = torch.randn(observation.shape[0], self.action_dim, device=observation.device)
            action = self.onestep_actor(observation,z)
            action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # (student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        action = noise.clone()
        t = torch.zeros(observation.shape[0], 1, device=observation.device)
        dt = 1.0 /self.flow_steps
        for _ in range(self.flow_steps):
            action = action + dt * self.bc_actor(observation,action,t)
            t = t+ dt
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # (student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        with torch.no_grad():
            z = torch.randn(next_observations.shape[0], self.action_dim, device=next_observations.device)
            next_action = self.onestep_actor(next_observations, z)
            next_action = torch.clamp(next_action, -1, 1)  
            q = self.target_critic(next_observations, next_action)  
            y = rewards + 0.5 * self.discount * (1.0 - dones.float()) * q.sum(dim = 0)

        loss = nn.functional.mse_loss(self.critic(observations,actions),y.unsqueeze(dim=0),reduction="none").mean(dim=-1).sum()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # (student): Compute the BC flow loss
        with torch.no_grad():
            z = torch.randn(observations.shape[0], self.action_dim, device=observations.device)
            t = torch.rand(observations.shape[0], 1, device=observations.device)  
            a = (1.0 - t)*z+t*actions
        loss = nn.functional.mse_loss(self.bc_actor(observations,a,t),actions-z)

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # (student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        with torch.no_grad():
            z = torch.randn(observations.shape[0], self.action_dim, device=observations.device)
            v_action = self.get_bc_action(observations,z)

        w_action = self.onestep_actor(observations,z)
        distill_loss = self.alpha * nn.functional.mse_loss(v_action,w_action)

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        w_action = torch.clamp(w_action, -1, 1)
        q = self.critic(observations,w_action)
        q = - q.mean(dim = 0)
        q_loss = q.mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = nn.functional.mse_loss(w_action,actions)

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # (student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 -self.target_update_rate) + param.data*self.target_update_rate)
