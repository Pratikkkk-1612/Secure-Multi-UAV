
import torch

from reinforcement_learning.masac import (
    ActorNetwork,
    CriticNetwork,
    update_actor,
)


def test_all_four_actor_updates():
    torch.manual_seed(42)

    n_agents = 4
    batch_size = 8

    # Four independent actors
    actors = [
        ActorNetwork() for _ in range(n_agents)
    ]

    # Two centralized critics per UAV
    critic_pairs = [
        (CriticNetwork(), CriticNetwork())
        for _ in range(n_agents)
    ]

    # Separate optimizer for each actor
    optimizers = [
        torch.optim.Adam(actor.parameters(), lr=3e-4)
        for actor in actors
    ]

    states = torch.randn(batch_size, 358)
    local_obs = torch.randn(batch_size, 4, 40)

    for i in range(n_agents):

        # Save all actor weights
        old_weights = [
            [p.detach().clone() for p in actor.parameters()]
            for actor in actors
        ]

        critic1, critic2 = critic_pairs[i]

        # Save critic weights
        old_critic_weights = [
            p.detach().clone()
            for critic in (critic1, critic2)
            for p in critic.parameters()
        ]

        # Update only actor i
        loss = update_actor(
            states,
            local_obs,
            actors,
            critic1,
            critic2,
            agent_index=i,
            actor_optimizer=optimizers[i],
            alpha=0.2,
        )

        assert torch.isfinite(torch.tensor(loss))

        # Check which actor changed
        for j, actor in enumerate(actors):
            changed = any(
                not torch.equal(old, p.detach())
                for old, p in zip(
                    old_weights[j],
                    actor.parameters()
                )
            )

            assert changed == (i == j)

        # Centralized critic weights must remain unchanged
        current_critic_params = [
            p
            for critic in (critic1, critic2)
            for p in critic.parameters()
        ]

        assert all(
            torch.equal(old, p.detach())
            for old, p in zip(
                old_critic_weights,
                current_critic_params
            )
        )

        print(f"UAV {i+1}: Actor update passed, loss={loss:.4f}")


def test_soft_target_update():

    from copy import deepcopy
    from reinforcement_learning.masac import soft_update

    critic = CriticNetwork()
    target = deepcopy(critic)

    target.requires_grad_(False)

    # Save original target weights
    old_target = [
        p.detach().clone()
        for p in target.parameters()
    ]

    # Simulate a change to the online critic
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(1.0)

    # Perform soft update
    tau = 0.005
    soft_update(target, critic, tau)

    for old, new, source in zip(
        old_target,
        target.parameters(),
        critic.parameters(),
    ):
        expected = (1 - tau) * old + tau * source

        assert torch.allclose(
            new, expected, atol=1e-6
        )

    assert all(
        not p.requires_grad
        for p in target.parameters()
    )

    print("Soft target update passed")


from reinforcement_learning.masac import (
    EntropyTemperature,
    update_temperature,
)


def test_entropy_temperature():

    temperature = EntropyTemperature(initial_alpha=0.2)

    optimizer = torch.optim.Adam(
        temperature.parameters(),
        lr=3e-4
    )

    old_alpha = temperature.alpha.item()

    # Illustrative log probabilities.
    # Continuous probability densities can have positive log values.
    log_probs = torch.full((8, 1), 4.0)

    new_alpha = update_temperature(
        temperature,
        log_probs,
        optimizer,
        target_entropy=-3.0,
    )

    assert new_alpha > 0.0
    assert new_alpha != old_alpha

    # With log_probs + target_entropy > 0,
    # this update should increase alpha.
    assert new_alpha > old_alpha

    print("Automatic entropy temperature passed")
    print("Old alpha:", old_alpha)
    print("New alpha:", new_alpha)


def test_masac_initialization():

    from reinforcement_learning.masac import MASAC

    masac = MASAC()

    assert len(masac.actors) == 4
    assert len(masac.critics1) == 4
    assert len(masac.critics2) == 4

    assert len(masac.target_critics1) == 4
    assert len(masac.target_critics2) == 4

    assert len(masac.temperatures) == 4

    assert len(masac.actor_optimizers) == 4
    assert len(masac.critic_optimizers) == 4
    assert len(masac.alpha_optimizers) == 4

    # Check target critic initialization
    for i in range(4):
        for online, target in [
            (masac.critics1[i], masac.target_critics1[i]),
            (masac.critics2[i], masac.target_critics2[i]),
        ]:
            for p, tp in zip(
                online.parameters(),
                target.parameters(),
            ):
                assert torch.equal(p, tp)
                assert not tp.requires_grad

    # Check actor and critic dimensions
    local_obs = torch.randn(8, 40)
    states = torch.randn(8, 358)
    joint_actions = torch.randn(8, 12)

    action, log_prob, _ = masac.actors[0](local_obs)
    q_value = masac.critics1[0](states, joint_actions)

    assert action.shape == (8, 3)
    assert log_prob.shape == (8, 1)
    assert q_value.shape == (8, 1)

    assert all(
        temperature.alpha.item() > 0
        for temperature in masac.temperatures
    )

    print("MASAC initialization passed")


def test_masac_training_update():
    import numpy as np

    from reinforcement_learning.masac import MASAC
    from reinforcement_learning.replay_buffer import ReplayBuffer

    rng = np.random.default_rng(42)

    buffer = ReplayBuffer(capacity=16, seed=42)
    agents = buffer.agent_names

    # Store 8 synthetic transitions
    for _ in range(8):
        buffer.add(
            state=rng.standard_normal(358).astype(np.float32),
            observations={
                a: rng.standard_normal(40).astype(np.float32)
                for a in agents
            },
            actions={
                a: rng.uniform(-1, 1, 3).astype(np.float32)
                for a in agents
            },
            rewards={
                a: float(rng.normal())
                for a in agents
            },
            next_state=rng.standard_normal(358).astype(np.float32),
            next_observations={
                a: rng.standard_normal(40).astype(np.float32)
                for a in agents
            },
            terminations={a: False for a in agents},
            truncations={a: False for a in agents},
        )

    masac = MASAC()

    # One complete MASAC training update
    metrics = masac.update(buffer, batch_size=8)

    assert metrics is not None

    for key in ("critic_loss", "actor_loss", "alpha"):
        assert len(metrics[key]) == 4
        assert np.isfinite(metrics[key]).all()

    assert all(a > 0 for a in metrics["alpha"])

    print("Complete MASAC training update passed")
    print(metrics)


def test_masac_action_selection():
    import numpy as np
    from reinforcement_learning.masac import MASAC

    masac = MASAC()

    observations = {
        f"uav_{i}": np.random.randn(40).astype(np.float32)
        for i in range(4)
    }

    # Training actions
    actions = masac.select_actions(
        observations,
        deterministic=False
    )

    assert len(actions) == 4

    for agent, action in actions.items():
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    # Deterministic evaluation
    actions1 = masac.select_actions(
        observations, deterministic=True
    )

    actions2 = masac.select_actions(
        observations, deterministic=True
    )

    for agent in observations:
        np.testing.assert_allclose(
            actions1[agent], actions2[agent]
        )

    print("MASAC action selection passed")


def test_masac_environment_integration():
    import numpy as np

    from environment import SAGINMultiUAVEnv
    from reinforcement_learning.masac import MASAC
    from reinforcement_learning.replay_buffer import ReplayBuffer

    env = SAGINMultiUAVEnv()
    masac = MASAC()
    buffer = ReplayBuffer(capacity=10, seed=42)

    try:
        # 1. Reset environment
        observations, _ = env.reset(seed=42)
        state = env.state().copy()

        assert state.shape == (358,)
        assert all(obs.shape == (40,)
                   for obs in observations.values())

        # 2. Generate four UAV actions
        actions = masac.select_actions(observations)

        # 3. Execute environment step
        next_obs, rewards, terminations, truncations, infos = (
            env.step(actions)
        )

        next_state = env.state().copy()

        # Use final observations if terminal step removes agents
        if set(next_obs) != set(env.possible_agents):
            next_obs = {
                agent: next_obs.get(
                    agent,
                    infos[agent].get("final_observation")
                )
                for agent in env.possible_agents
            }

        # 4. Store real experience
        buffer.add(
            state=state,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_state=next_state,
            next_observations=next_obs,
            terminations=terminations,
            truncations=truncations,
        )

        assert len(buffer) == 1

        # 5. Perform one MASAC learning update
        metrics = masac.update(buffer, batch_size=1)

        assert metrics is not None

        for values in metrics.values():
            assert np.isfinite(values).all()

        print("Environment + MASAC integration passed")

    finally:
        env.close()

