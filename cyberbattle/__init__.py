# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Initialize CyberBattleSim module"""

# Prefer the classic `gym` registry so that `gym.make` finds our envs.
# Fall back to `gymnasium` if `gym` isn't available.
try:
    from gym.envs.registration import registry, EnvSpec
    from gym.error import Error
except Exception:
    from gymnasium.envs.registration import registry, EnvSpec
    from gymnasium.error import Error

from . import simulation
from . import agents
from ._env.cyberbattle_env import AttackerGoal, DefenderGoal
from .samples.chainpattern import chainpattern
from .samples.toyctf import toy_ctf
from .samples.active_directory import generate_ad
from .simulation import generate_network, model

__all__ = (
    "simulation",
    "agents",
)


def register(id: str, cyberbattle_env_identifiers: model.Identifiers, **kwargs):
    """Register an environment in whichever registry is available.

    This supports both `gym` (which exposes `registry.env_specs`) and
    `gymnasium` (which exposes a mapping-like `registry`). Attach
    CyberBattle-specific metadata to the created EnvSpec.
    """
    env_specs_store = getattr(registry, "env_specs", None)
    if env_specs_store is not None:
        if id in env_specs_store:
            raise Error("Cannot re-register id: {}".format(id))
        spec = EnvSpec(id, **kwargs)
        spec.ports = cyberbattle_env_identifiers.ports
        spec.properties = cyberbattle_env_identifiers.properties
        spec.local_vulnerabilities = cyberbattle_env_identifiers.local_vulnerabilities
        spec.remote_vulnerabilities = cyberbattle_env_identifiers.remote_vulnerabilities
        registry.env_specs[id] = spec
    else:
        if id in registry:
            raise Error("Cannot re-register id: {}".format(id))
        spec = EnvSpec(id, **kwargs)
        spec.ports = cyberbattle_env_identifiers.ports
        spec.properties = cyberbattle_env_identifiers.properties
        spec.local_vulnerabilities = cyberbattle_env_identifiers.local_vulnerabilities
        spec.remote_vulnerabilities = cyberbattle_env_identifiers.remote_vulnerabilities
        registry[id] = spec


def _unregister_if_present(env_id: str):
    env_specs_store = getattr(registry, "env_specs", None)
    if env_specs_store is not None:
        if env_id in env_specs_store:
            del env_specs_store[env_id]
    else:
        if env_id in registry:
            del registry[env_id]


_unregister_if_present("CyberBattleToyCtf-v0")

register(
    id="CyberBattleToyCtf-v0",
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_toyctf:CyberBattleToyCtf",
    kwargs={"defender_agent": None, "attacker_goal": AttackerGoal(own_atleast=6), "defender_goal": DefenderGoal(eviction=True)},
    # max_episode_steps=2600,
)

_unregister_if_present("CyberBattleTiny-v0")
register(
    id="CyberBattleTiny-v0",
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_tiny:CyberBattleTiny",
    kwargs={"defender_agent": None, "attacker_goal": AttackerGoal(own_atleast=6), "defender_goal": DefenderGoal(eviction=True), "maximum_total_credentials": 10, "maximum_node_count": 10},
    # max_episode_steps=2600,
)


_unregister_if_present("CyberBattleRandom-v0")
register(
    id="CyberBattleRandom-v0",
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_random:CyberBattleRandom",
)

_unregister_if_present("CyberBattleChain-v0")
register(
    id="CyberBattleChain-v0",
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_chain:CyberBattleChain",
    kwargs={"size": 4, "defender_agent": None, "attacker_goal": AttackerGoal(own_atleast_percent=1.0), "defender_goal": DefenderGoal(eviction=True), "winning_reward": 5000.0, "losing_reward": 0.0},
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectory-v{i}" for i in range(0, 10)]
for index, env in enumerate(ad_envs):
    _unregister_if_present(env)

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point="cyberbattle._env.active_directory:CyberBattleActiveDirectory",
        kwargs={
            "seed": index,
            "maximum_discoverable_credentials_per_action": 50000,
            "maximum_node_count": 30,
            "maximum_total_credentials": 50000,
        },
    )

_unregister_if_present("ActiveDirectoryTiny-v0")
register(
    id="ActiveDirectoryTiny-v0",
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.active_directory:CyberBattleActiveDirectoryTiny",
    kwargs={"maximum_discoverable_credentials_per_action": 50000, "maximum_node_count": 30, "maximum_total_credentials": 50000},
)
