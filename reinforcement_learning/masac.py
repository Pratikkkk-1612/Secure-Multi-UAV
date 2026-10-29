import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy


class ActorNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int = 40,
        action_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):

        x = self.net(obs)

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mu)

        return action, log_prob, deterministic_action
class CriticNetwork(nn.Module):

    def __init__(
        self,
        state_dim=358,
        joint_action_dim=12,
        hidden_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, joint_action):

        x = torch.cat([state, joint_action], dim=-1)

        return self.net(x)

@torch.no_grad()
def compute_critic_target(
    next_states,
    next_local_obs,
    rewards,
    dones,
    actors,
    target_critic1,
    target_critic2,
    agent_index,
    gamma=0.99,
    alpha=0.2,
):

    next_actions = []
    own_log_prob = None

    # Sample next action from every UAV actor
    for j, actor in enumerate(actors):

        action, log_prob, _ = actor(
            next_local_obs[:, j, :]
        )

        next_actions.append(action)

        if j == agent_index:
            own_log_prob = log_prob

    # Combine all four actions: (batch, 12)
    joint_next_actions = torch.cat(
        next_actions, dim=-1
    )

    # Calculate target Q-values
    q1_next = target_critic1(
        next_states, joint_next_actions
    )

    q2_next = target_critic2(
        next_states, joint_next_actions
    )

    min_q_next = torch.min(q1_next, q2_next)

    # Entropy-adjusted future value
    future_value = min_q_next - alpha * own_log_prob

    # Bellman target
    target = (
        rewards[:, agent_index:agent_index + 1]
        + gamma
        * (1.0 - dones[:, agent_index:agent_index + 1].float())
        * future_value
    )

    return target

def compute_critic_loss(
    states,
    actions,
    targets,
    critic1,
    critic2,
):
    # Combine 4 UAV actions into 12 values
    joint_actions = actions.reshape(actions.shape[0], -1)

    # Both critics predict Q-values
    q1 = critic1(states, joint_actions)
    q2 = critic2(states, joint_actions)

    # Compare predictions with Bellman target
    loss1 = F.mse_loss(q1, targets)
    loss2 = F.mse_loss(q2, targets)

    return loss1 + loss2

def update_critics(
    states,
    actions,
    targets,
    critic1,
    critic2,
    critic_optimizer,
):
    # Step 1: Calculate twin-critic loss
    loss = compute_critic_loss(
        states, actions, targets,
        critic1, critic2
    )

    # Step 2: Clear previous gradients
    critic_optimizer.zero_grad(set_to_none=True)

    # Step 3: Calculate new gradients
    loss.backward()

    # Step 4: Prevent excessively large gradients
    torch.nn.utils.clip_grad_norm_(
        list(critic1.parameters()) +
        list(critic2.parameters()),
        max_norm=5.0
    )

    # Step 5: Update critic weights
    critic_optimizer.step()

    return float(loss.item())


def compute_actor_loss(
    states,
    local_obs,
    actors,
    critic1,
    critic2,
    agent_index,
    alpha=0.2,
):
    joint_actions = []
    own_log_prob = None

    # Sample actions from all four UAVs
    for j, actor in enumerate(actors):

        if j == agent_index:
            # Our UAV: preserve gradients
            action, log_prob, _ = actor(
                local_obs[:, j, :]
            )
            own_log_prob = log_prob

        else:
            # Other UAVs: sample without updating them
            with torch.no_grad():
                action, _, _ = actor(
                    local_obs[:, j, :]
                )

        joint_actions.append(action)

    # Combine all actions: (batch, 12)
    joint_actions = torch.cat(
        joint_actions, dim=-1
    )

    # Evaluate joint actions using centralized critics
    q1 = critic1(states, joint_actions)
    q2 = critic2(states, joint_actions)

    min_q = torch.min(q1, q2)

    # SAC actor objective
    actor_loss = (
        alpha * own_log_prob - min_q
    ).mean()

    return actor_loss


def update_actor(
    states,
    local_obs,
    actors,
    critic1,
    critic2,
    agent_index,
    actor_optimizer,
    alpha=0.2,
):
    # Freeze critic weights during actor training
    critic_params = (
        list(critic1.parameters())
        + list(critic2.parameters())
    )

    original_grad_flags = [
        p.requires_grad for p in critic_params
    ]

    try:
        for p in critic_params:
            p.requires_grad_(False)

        # Clear previous actor gradients
        actor_optimizer.zero_grad(set_to_none=True)

        # Calculate actor loss
        loss = compute_actor_loss(
            states,
            local_obs,
            actors,
            critic1,
            critic2,
            agent_index,
            alpha,
        )

        # Backpropagate through selected actor
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            actors[agent_index].parameters(),
            max_norm=5.0,
        )

        # Update actor weights
        actor_optimizer.step()

    finally:
        # Restore critic gradient settings
        for p, flag in zip(
            critic_params, original_grad_flags
        ):
            p.requires_grad_(flag)

    return float(loss.item())


@torch.no_grad()
def soft_update(target_network, source_network, tau=0.005):

    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be between 0 and 1")

    for target_param, source_param in zip(
        target_network.parameters(),
        source_network.parameters(),
    ):
        target_param.mul_(1.0 - tau)
        target_param.add_(tau * source_param)


class EntropyTemperature(nn.Module):

    def __init__(self, initial_alpha=0.2):
        super().__init__()

        if initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")

        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(float(initial_alpha)))
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()


def update_temperature(
    temperature,
    log_probs,
    optimizer,
    target_entropy=-3.0,
):
    # Log probabilities must not update the actor here
    alpha_loss = -(
        temperature.log_alpha
        * (log_probs.detach() + target_entropy)
    ).mean()

    optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    optimizer.step()

    return float(temperature.alpha.detach().item())


class MASAC(nn.Module):

    def __init__(
        self,
        n_agents=4,
        state_dim=358,
        local_obs_dim=40,
        action_dim=3,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4,
        initial_alpha=0.2,
        device="cpu",
    ):
        super().__init__()

        self.n_agents = n_agents
        self.agent_names = tuple(f"uav_{i}" for i in range(n_agents))
        self.local_obs_dim = local_obs_dim

        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -float(action_dim)
        self.device = torch.device(device)

        joint_action_dim = n_agents * action_dim

        # 1. Independent stochastic actors
        self.actors = nn.ModuleList([
            ActorNetwork(
                obs_dim=local_obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_agents)
        ])

        # 2. Two centralized critics per UAV
        self.critics1 = nn.ModuleList([
            CriticNetwork(
                state_dim=state_dim,
                joint_action_dim=joint_action_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_agents)
        ])

        self.critics2 = nn.ModuleList([
            CriticNetwork(
                state_dim=state_dim,
                joint_action_dim=joint_action_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_agents)
        ])

        # 3. Target critics start with identical weights
        self.target_critics1 = deepcopy(self.critics1)
        self.target_critics2 = deepcopy(self.critics2)

        self.target_critics1.requires_grad_(False)
        self.target_critics2.requires_grad_(False)

        # 4. Independent entropy temperatures
        self.temperatures = nn.ModuleList([
            EntropyTemperature(initial_alpha)
            for _ in range(n_agents)
        ])

        # Move networks to selected device
        self.to(self.device)

        # 5. Independent actor optimizers
        self.actor_optimizers = [
            torch.optim.Adam(
                actor.parameters(),
                lr=learning_rate,
            )
            for actor in self.actors
        ]

        # 6. One optimizer for each twin-critic pair
        self.critic_optimizers = [
            torch.optim.Adam(
                list(self.critics1[i].parameters())
                + list(self.critics2[i].parameters()),
                lr=learning_rate,
            )
            for i in range(n_agents)
        ]

        # 7. Independent temperature optimizers
        self.alpha_optimizers = [
            torch.optim.Adam(
                self.temperatures[i].parameters(),
                lr=learning_rate,
            )
            for i in range(n_agents)
        ]


    def update(self, replay_buffer, batch_size=64):

        # Do not train until sufficient experiences exist
        if not replay_buffer.can_sample(batch_size):
            return None

        # 1. Sample experiences
        batch = replay_buffer.sample(batch_size)

        states = batch["states"].to(self.device)
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)

        next_states = batch["next_states"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        metrics = {
            "critic_loss": [],
            "actor_loss": [],
            "alpha": [],
        }

        # 2. Train each UAV's actor and centralized critics
        for i in range(self.n_agents):

            alpha = self.temperatures[i].alpha.detach()

            # A. Calculate Bellman target
            target = compute_critic_target(
                next_states=next_states,
                next_local_obs=next_observations,
                rewards=rewards,
                dones=dones,
                actors=self.actors,
                target_critic1=self.target_critics1[i],
                target_critic2=self.target_critics2[i],
                agent_index=i,
                gamma=self.gamma,
                alpha=alpha,
            )

            # B. Update online twin critics
            critic_loss = update_critics(
                states,
                actions,
                target,
                self.critics1[i],
                self.critics2[i],
                self.critic_optimizers[i],
            )

            # C. Update selected actor
            actor_loss = update_actor(
                states=states,
                local_obs=observations,
                actors=self.actors,
                critic1=self.critics1[i],
                critic2=self.critics2[i],
                agent_index=i,
                actor_optimizer=self.actor_optimizers[i],
                alpha=alpha,
            )

            # D. Update entropy temperature
            with torch.no_grad():
                _, log_prob, _ = self.actors[i](
                    observations[:, i, :]
                )

            new_alpha = update_temperature(
                temperature=self.temperatures[i],
                log_probs=log_prob,
                optimizer=self.alpha_optimizers[i],
                target_entropy=self.target_entropy,
            )

            # E. Soft-update target critics
            soft_update(
                self.target_critics1[i],
                self.critics1[i],
                self.tau,
            )

            soft_update(
                self.target_critics2[i],
                self.critics2[i],
                self.tau,
            )

            metrics["critic_loss"].append(critic_loss)
            metrics["actor_loss"].append(actor_loss)
            metrics["alpha"].append(new_alpha)

        return metrics

    
    @torch.no_grad()
    def select_actions(self, observations, deterministic=False):

        if set(observations) != set(self.agent_names):
            raise ValueError("Observations must contain all UAV agents")

        actions = {}

        for i, agent in enumerate(self.agent_names):

            obs = np.asarray(
                observations[agent],
                dtype=np.float32
            )

            if obs.shape != (self.local_obs_dim,):
                raise ValueError(
                    f"Invalid observation shape for {agent}"
                )

            if not np.isfinite(obs).all():
                raise ValueError("Observation contains NaN or Inf")

            # Convert local observation into a batch of size 1
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            # Generate stochastic and deterministic actions
            sampled_action, _, mean_action = (
                self.actors[i](obs_tensor)
            )

            action = (
                mean_action if deterministic
                else sampled_action
            )

            # Convert back into NumPy
            actions[agent] = np.clip(
                action.squeeze(0).cpu().numpy(),
                -1.0,
                1.0,
            ).astype(np.float32)

        return actions
