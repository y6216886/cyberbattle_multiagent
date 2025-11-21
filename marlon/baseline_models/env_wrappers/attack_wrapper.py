import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from plotly.missing_ipywidgets import FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gym
from gym import spaces
from gym.spaces.space import Space

from cyberbattle._env.cyberbattle_env import Action, CyberBattleEnv, EnvironmentBounds, Observation
from cyberbattle.simulation import commandcontrol, model

from marlon.baseline_models.env_wrappers.environment_event_source import IEnvironmentObserver, EnvironmentEventSource
from marlon.baseline_models.env_wrappers.reward_store import IRewardStore

class AttackerEnvWrapper(gym.Env, IRewardStore, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for stablebaselines-3 models to learn how to attack."""

    nested_spaces = ['credential_cache_matrix', 'leaked_credentials']
    other_removed_spaces = ['connect']
    int32_spaces = ['customer_data_found', 'escalation', 'lateral_move', 'newly_discovered_nodes_count', 'probe_result']

    def __init__(self,
        cyber_env: CyberBattleEnv,
        event_source: Optional[EnvironmentEventSource] = None,
        max_timesteps=2000,
        invalid_action_reward_modifier=-1,
        invalid_action_reward_multiplier=1,
        loss_reward=-5000):

        super().__init__()
        self.cyber_env: CyberBattleEnv = cyber_env
        self.bounds: EnvironmentBounds = self.cyber_env.bounds
        self.max_timesteps = max_timesteps
        self.invalid_action_reward_modifier = invalid_action_reward_modifier
        self.invalid_action_reward_multiplier = invalid_action_reward_multiplier
        self.loss_reward = loss_reward

        # These should be set during reset()
        self.timesteps = None
        self.cyber_rewards = [] # The rewards as given by CyberBattle, before modification.
        self.rewards = [] # The rewards returned by this wrapper, after modification.

        # Access protected members only in the ctor to avoid pylint warnings.
        self.node_count = cyber_env._CyberBattleEnv__node_count
        self.__get_privilegelevel_array = cyber_env._CyberBattleEnv__get_privilegelevel_array

        self.observation_space: Space = self.__create_observation_space(cyber_env)
        self.action_space: Space = self.__create_action_space(cyber_env)

        self.valid_action_count = 0
        self.invalid_action_count = 0

        # Add this object as an observer of the cyber env.
        if event_source is None:
            event_source = EnvironmentEventSource()

        self.event_source = event_source
        event_source.add_observer(self)

        self.reset_request = False

    def __create_observation_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        # Copy the env's internal spaces mapping. We'll create new spaces
        # using the same spaces implementation (gym or gymnasium) as the env
        # to avoid mixing Space classes which causes isinstance checks to fail.
        observation_space = cyber_env.observation_space.__dict__['spaces'].copy()

        # Use `gym.spaces` to construct observation spaces so they are
        # compatible with Stable-Baselines3 (which expects `gym` types).
        import gym.spaces as gspaces  # type: ignore

        # Helper: convert gymnasium.Space instances to gym.spaces equivalents
        def convert_space(space):
            # If already a gym space, return as-is
            if isinstance(space, gspaces.Space):
                return space

            # Try to detect gymnasium spaces by module name
            mod = getattr(space.__class__, '__module__', '')
            name = getattr(space.__class__, '__name__', '')
            try:
                import gymnasium.spaces as gns
            except Exception:
                gns = None

            if gns is not None and isinstance(space, gns.Space):
                # Map common space types
                if name == 'Discrete':
                    return gspaces.Discrete(space.n)
                if name == 'MultiBinary':
                    # gymnasium MultiBinary may expose `n` or `shape`
                    if hasattr(space, 'n'):
                        return gspaces.MultiBinary(space.n)
                    if hasattr(space, 'shape'):
                        # flatten shape to a single dimension
                        import numpy as _np
                        return gspaces.MultiBinary(int(_np.prod(space.shape)))
                if name == 'MultiDiscrete':
                    import numpy as _np
                    nvec = _np.array(space.nvec, copy=True)
                    return gspaces.MultiDiscrete(nvec)
                if name == 'Box':
                    # preserve bounds and dtype
                    low = getattr(space, 'low', None)
                    high = getattr(space, 'high', None)
                    dtype = getattr(space, 'dtype', None)
                    return gspaces.Box(low=low, high=high, dtype=dtype)
                if name == 'Tuple':
                    return gspaces.Tuple([convert_space(s) for s in space.spaces])
                if name == 'Dict':
                    return gspaces.Dict({k: convert_space(v) for k, v in space.spaces.items()})

            # Unknown/unsupported space object: if it's a numpy array-like, convert to Box
            try:
                import numpy as _np
                if isinstance(space, (_np.ndarray, list, tuple)):
                    arr = _np.asarray(space)
                    low = _np.full(arr.shape, -_np.inf)
                    high = _np.full(arr.shape, _np.inf)
                    return gspaces.Box(low=low, high=high, dtype=arr.dtype)
            except Exception:
                pass

            # As a last resort, raise an informative error so developers can adapt
            raise TypeError(f"Cannot convert observation space of type {type(space)} to gym.spaces equivalent")

        # Flatten the action_mask field.
        observation_space['local_vulnerability'] = observation_space['action_mask']['local_vulnerability']
        observation_space['remote_vulnerability'] = observation_space['action_mask']['remote_vulnerability']
        observation_space['connect'] = observation_space['action_mask']['connect']
        del observation_space['action_mask']

        # Change action_mask spaces to use node count instead of maximum node count.
        observation_space['local_vulnerability'] = gspaces.MultiBinary(self.node_count * self.bounds.local_attacks_count)
        observation_space['remote_vulnerability'] = gspaces.MultiBinary(self.node_count * self.node_count * self.bounds.remote_attacks_count)

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        # Use pop(default) to be robust across different cyberbattle versions.
        observation_space.pop('credential_cache', None)
        observation_space.pop('discovered_nodes', None)
        observation_space.pop('explored_network', None)
        observation_space.pop('_discovered_nodes', None)
        observation_space.pop('_explored_network', None)

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation_space[space]

        # This is incorrectly set to spaces.MultiBinary(2)
        # It's a single value in the returned observations
        observation_space['customer_data_found'] = gspaces.Discrete(2)

        # This is incorrectly set to spaces.MultiDiscrete(model.PrivilegeLevel.MAXIMUM + 1), when it is only one value
        observation_space['escalation'] = gspaces.Discrete(model.PrivilegeLevel.MAXIMUM + 1)

        # Convert any gymnasium space instances in the dict to gym.spaces equivalents
        converted = {}
        for k, v in observation_space.items():
            converted[k] = convert_space(v)

        return gspaces.Dict(converted)

    def __create_action_space(self, cyber_env: CyberBattleEnv) -> gym.Space:
        self.action_subspaces = {}
        # First action defines which action subspace to use
        # local_vulnerability, remote_vulnerability, or connect
        action_space = [3]

        # CyberBattle's action space is a dict of nested action spaces.
        # We need to flatten it into a single multidiscrete and keep
        # track of which values correspond to which nested values so
        # we can reconstruct the action later.
        subspace_index = 0
        # Use `gym.spaces` to construct action spaces to ensure compatibility
        # with Stable-Baselines3 (avoid gymnasium space classes here).
        import gym.spaces as gspaces  # type: ignore

        for (key, value) in cyber_env.action_space.spaces.items():
            subspace_start = len(action_space)
            for vec in value.nvec:
                action_space.append(vec)

            # Action subspace takes the form:
            # [('subspace_name', 1, 3), ('subspace_name2', 3, 5)]
            self.action_subspaces[subspace_index] = (key, subspace_start, len(action_space))
            subspace_index += 1

        return gspaces.MultiDiscrete(action_space)

    def __get_owned_nodes(self):
        return np.nonzero(self.__get_privilegelevel_array())[0]

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # The first action value corresponds to the subspace
        action_subspace = self.action_subspaces[action[0]]

        # Translate the flattened action back into the nested
        # subspace action for CyberBattle. It takes the form:
        # {'subspace_name': [0, 1, 2]}
        translated_action = {action_subspace[0]: action[action_subspace[1]:action_subspace[2]]}

        # For reference:
        # ```python
        # action_spaces: ActionSpaceDict = {
        #     "local_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, vulnerability_id
        #         [maximum_node_count, local_vulnerabilities_count]),
        #     "remote_vulnerability": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, vulnerability_id
        #         [maximum_node_count, maximum_node_count, remote_vulnerabilities_count]),
        #     "connect": spaces.MultiDiscrete(
        #         # source_node_id, target_node_id, target_port, credential_id
        #         # (by index of discovery: 0 for initial node, 1 for first discovered node, ...)
        #         [maximum_node_count, maximum_node_count, port_count, maximum_total_credentials])
        # }
        # ```

        # First, check if the action is valid.
        reward_modifier = 0
        if not self.cyber_env.is_action_valid(translated_action):
            # If it is not valid, we will try picking a random valid node and hoping
            # that makes the action valid.

            # Pick source node at random (owned and with the desired feature encoding)
            potential_source_nodes = [
                from_node
                for from_node in self.__get_owned_nodes()
                #if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))
            ]

            if len(potential_source_nodes) > 0:
                source_node = np.random.choice(potential_source_nodes)

                if action_subspace[0] == 'local_vulnerability':
                    # Replace node from the algorithm with a valid node.
                    translated_action[action_subspace[0]][0] = source_node
                else:
                    # Target node can be any potential node excluding source node.
                    potential_source_nodes.remove(source_node)

                    if len(potential_source_nodes) > 0:
                        target_node = np.random.choice(potential_source_nodes)

                        # Replace source and target node from the algorithm with valid nodes.
                        translated_action[action_subspace[0]][0] = source_node
                        translated_action[action_subspace[0]][1] = target_node
                    else:
                        # No potential target nodes
                        pass
            else:
                # No potential source nodes
                pass

        # If the action is still invalid, sample a random valid action.
        # TODO: Try invalid action masks instead of sampling a random valid action; 'Dynamic action spaces'.
        # https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
        is_invalid = False
        if not self.cyber_env.is_action_valid(translated_action):
            # sample local and remote actions only (excludes connect action)
            translated_action = self.cyber_env.sample_valid_action(kinds=[0, 1, 2])
            self.invalid_action_count += 1

            reward_modifier += self.invalid_action_reward_modifier
            is_invalid = True
        else:
            self.valid_action_count += 1

        # Call the underlying env.step and support both gym and gymnasium return
        # signatures. Normalize to (observation, reward, done, info).
        step_result = self.cyber_env.step(translated_action)
        # step_result can be (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            observation, reward, done, info = step_result
        else:
            # Fallback: try unpacking defensively
            try:
                observation = step_result[0]
                reward = step_result[1]
                done = step_result[2]
                info = step_result[-1]
            except Exception:
                raise RuntimeError("Unsupported env.step return signature")

        transformed_observation = self.transform_observation(observation)
        self.cyber_rewards.append(reward)

        if done:
            logging.warning("Attacker Won")

        self.timesteps += 1
        if self.reset_request:
            done = True

        if self.timesteps > self.max_timesteps:
            done = True

        # If action was invalid, multiplier is applied before reward modifier
        if is_invalid:
            reward = reward * self.invalid_action_reward_multiplier

        reward += reward_modifier
        self.rewards.append(reward)

        return transformed_observation, reward, done, info

    def reset(self) -> Observation:
        logging.debug('Reset Attacker')
        if not self.reset_request:
            last_reward = self.rewards[-1] if len(self.rewards) > 0 else 0
            self.event_source.notify_reset(last_reward)

        # Support both gym (obs) and gymnasium (obs, info) reset return signatures.
        reset_result = self.cyber_env.reset()
        if isinstance(reset_result, tuple) or isinstance(reset_result, list):
            # gymnasium returns (obs, info)
            observation = reset_result[0]
        else:
            observation = reset_result

        self.reset_request = False
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.timesteps = 0
        self.cyber_rewards = []
        self.rewards = []

        return self.transform_observation(observation)

    def on_reset(self, last_rewards):
        logging.info('on_reset Attacker')
        self.reset_request = True

    def transform_observation(self, observation) -> Observation:
        # Flatten the action_mask field
        observation['local_vulnerability'] = observation['action_mask']['local_vulnerability']
        observation['remote_vulnerability'] = observation['action_mask']['remote_vulnerability']
        observation['connect'] = observation['action_mask']['connect']
        del observation['action_mask']

        # TODO: Retain real values
        #if observation['credential_cache_matrix'].shape == (1,2):
        credential_cache_matrix = []
        for _ in range(self.bounds.maximum_total_credentials):
            credential_cache_matrix.append(np.zeros((2,)))

        # TODO: Clean this up a bit, action masks are not needed here
        observation['credential_cache_matrix'] = tuple(credential_cache_matrix)
        observation['discovered_nodes_properties'] = np.zeros((self.bounds.maximum_node_count * self.bounds.property_count,))
        observation['nodes_privilegelevel'] = np.zeros((self.bounds.maximum_node_count,))

        # Flatten action_mask subspaces
        # local_vulnerability comes in shape (node_count, local_attacks_count,)
        # but needs to be (node_count * local_attacks_count,)
        local_vulnerability = np.zeros((self.node_count * self.bounds.local_attacks_count,))
        flat_index = 0
        for i in range(self.node_count):
            for j in range(self.bounds.local_attacks_count):
                local_vulnerability[flat_index] = observation['local_vulnerability'][i][j]
                flat_index += 1
        observation['local_vulnerability'] = local_vulnerability

        # remote_vulnerability comes in shape (node_count, node_count, remote_attacks_count,)
        # but needs to be (node_count * node_count * remote_attacks_count,)
        remote_vulnerability = np.zeros((self.node_count * self.node_count * self.bounds.remote_attacks_count,))
        flat_index = 0
        for i in range(self.node_count):
            for j in range(self.node_count):
                for k in range(self.bounds.local_attacks_count):
                    remote_vulnerability[flat_index] = observation['remote_vulnerability'][i][j][k]
                    flat_index += 1
        observation['remote_vulnerability'] = remote_vulnerability

        # Remove 'info' fields added by cyberbattle that do not represent algorithm inputs
        observation.pop('credential_cache', None)
        observation.pop('discovered_nodes', None)
        observation.pop('explored_network', None)
        observation.pop('_discovered_nodes', None)
        observation.pop('_explored_network', None)

        # Stable baselines does not like numpy wrapped ints
        for space in self.int32_spaces:
            observation[space] = int(observation[space])

        # TODO: Reformat these spaces so they don't have to be removed
        # Remove nested Tuple/Dict spaces
        for space in self.nested_spaces + self.other_removed_spaces:
            del observation[space]

        return observation

    def close(self) -> None:
        return self.cyber_env.close()

    def render(self, mode: str = 'human') -> None:
        return self.cyber_env.render(mode)

    def render_as_fig(self, print_attacks=False) -> FigureWidget:
        # NOTE: This method is exactly the same as CyberBattleEnv.render_as_fig() except where noted.

        debug = commandcontrol.EnvironmentDebugging(self.cyber_env._actuator)
        # CHANGE: Parameter to decide whether to print this.
        if print_attacks:
            self.cyber_env._actuator.print_all_attacks()

        # plot the cumulative reward and network side by side using plotly
        fig = make_subplots(rows=1, cols=2)

        # CHANGE: Uses this environment's rewards instead of CyberBattle's.
        fig.add_trace(go.Scatter(y=np.array(self.cyber_rewards).cumsum(),
            name='cumulative reward'), row=1, col=1)

        traces, layout = debug.network_as_plotly_traces(xref="x2", yref="y2")
        for trace in traces:
            fig.add_trace(trace, row=1, col=2)
        fig.update_layout(layout)
        return fig

    @property
    def episode_rewards(self) -> List[float]:
        return self.cyber_rewards
