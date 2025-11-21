"""Flatten a Dict observation space into a single Box and flatten observations.
This wrapper converts each supported subspace (Discrete, MultiBinary, MultiDiscrete, Box)
into a flattened numeric array and concatenates them into a 1D float32 Box.
"""
from gym import ObservationWrapper, spaces
import numpy as np
from typing import OrderedDict
import math

class FlattenToBoxWrapper(ObservationWrapper):
    def __init__(self, env, ignore_fields=None):
        super().__init__(env)
        if ignore_fields is None:
            ignore_fields = ['action_mask']
        self.ignore_fields = set(ignore_fields)

        orig_space = env.observation_space
        if not isinstance(orig_space, spaces.Dict):
            # Nothing to do
            self.observation_space = orig_space
            return

        # Compute flattened subspaces in order
        self.keys = []
        self.subspaces = []
        total_dim = 0
        for k, sub in orig_space.spaces.items():
            if k in self.ignore_fields:
                continue
            # Only support certain space types
            if isinstance(sub, spaces.Discrete):
                dim = 1
            elif isinstance(sub, spaces.MultiBinary):
                n = getattr(sub, 'n', None)
                # n can be an int or an array-like; coerce to array and multiply elements
                n_arr = np.asarray(n)
                if n_arr.size == 0:
                    dim = 0
                else:
                    dim = int(math.prod(int(x) for x in n_arr.ravel()))
            elif isinstance(sub, spaces.MultiDiscrete):
                nvec = getattr(sub, 'nvec', None)
                # MultiDiscrete is a vector of discrete variables; flattened length = number of entries
                nvec_arr = np.asarray(nvec)
                dim = int(nvec_arr.size)
            elif isinstance(sub, spaces.Box):
                dim = int(math.prod([int(x) for x in sub.shape]))
            elif isinstance(sub, spaces.Tuple):
                # Sum dims of tuple elements
                dim = 0
                for s in sub.spaces:
                    if isinstance(s, spaces.Discrete):
                        dim += 1
                    elif isinstance(s, spaces.MultiBinary):
                        n = getattr(s, 'n', None)
                        n_arr = np.asarray(n)
                        if n_arr.size == 0:
                            dim += 0
                        else:
                            dim += int(math.prod(int(x) for x in n_arr.ravel()))
                    elif isinstance(s, spaces.MultiDiscrete):
                        nvec = getattr(s, 'nvec', None)
                        nvec_arr = np.asarray(nvec)
                        dim += int(nvec_arr.size)
                    elif isinstance(s, spaces.Box):
                        dim += int(math.prod([int(x) for x in s.shape]))
                    else:
                        raise NotImplementedError(f"Unsupported tuple subspace type: {type(s)}")
            elif isinstance(sub, spaces.Dict):
                # flatten nested dict by summing its subspaces
                dim = 0
                for _, s in sub.spaces.items():
                    if isinstance(s, spaces.Discrete):
                        dim += 1
                    elif isinstance(s, spaces.MultiBinary):
                        n = getattr(s, 'n', None)
                        n_arr = np.asarray(n)
                        if n_arr.size == 0:
                            dim += 0
                        else:
                            dim += int(math.prod(int(x) for x in n_arr.ravel()))
                    elif isinstance(s, spaces.MultiDiscrete):
                        nvec = getattr(s, 'nvec', None)
                        nvec_arr = np.asarray(nvec)
                        dim += int(nvec_arr.size)
                    elif isinstance(s, spaces.Box):
                        dim += int(math.prod([int(x) for x in s.shape]))
                    else:
                        raise NotImplementedError(f"Unsupported dict nested subspace type: {type(s)}")
            else:
                raise NotImplementedError(f"Unsupported observation subspace type: {type(sub)} for key {k}")

            self.keys.append(k)
            self.subspaces.append(sub)
            total_dim += dim

        # Avoid allocating huge arrays for low/high when total_dim is large
        # Use scalar bounds with explicit shape to remain memory efficient.
        # This works with both Gym and Gymnasium Box constructors.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def _flatten_value(self, subspace, value):
        # Convert value to 1D numpy array
        if isinstance(subspace, spaces.Discrete):
            return np.asarray([int(value)], dtype=np.float32)
        if isinstance(subspace, spaces.MultiBinary):
            arr = np.asarray(value, dtype=np.float32)
            return arr.reshape(-1)
        if isinstance(subspace, spaces.MultiDiscrete):
            arr = np.asarray(value, dtype=np.float32)
            return arr.reshape(-1)
        if isinstance(subspace, spaces.Box):
            arr = np.asarray(value, dtype=np.float32)
            return arr.reshape(-1)
        if isinstance(subspace, spaces.Tuple):
            parts = []
            for s, v in zip(subspace.spaces, value):
                parts.append(self._flatten_value(s, v))
            return np.concatenate(parts).reshape(-1)
        if isinstance(subspace, spaces.Dict):
            parts = []
            for sk, sv in subspace.spaces.items():
                parts.append(self._flatten_value(sv, value[sk]))
            return np.concatenate(parts).reshape(-1)
        raise NotImplementedError(f"Unsupported subspace type for flattening: {type(subspace)}")

    def observation(self, observation):
        if not isinstance(self.env.observation_space, spaces.Dict):
            return observation
        parts = []
        for k, sub in zip(self.keys, self.subspaces):
            if k in self.ignore_fields:
                continue
            v = observation[k]
            parts.append(self._flatten_value(sub, v))
        if parts:
            return np.concatenate(parts).astype(np.float32)
        else:
            return np.array([], dtype=np.float32)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        # gym vs gymnasium reset signature differences
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
            return self.observation(obs), info
        else:
            return self.observation(res)

    def step(self, action):
        res = self.env.step(action)
        # handle gymnasium (obs, reward, terminated, truncated, info)
        if isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = bool(terminated or truncated)
            return self.observation(obs), reward, done, info
        elif isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
            return self.observation(obs), reward, done, info
        else:
            # defensive unpack
            obs = res[0]
            reward = res[1]
            done = res[2]
            info = res[-1]
            return self.observation(obs), reward, done, info
