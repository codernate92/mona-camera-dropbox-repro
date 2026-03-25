"""Tests for PPO observation encoding and MONA rollout alignment."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from approval_spectrum.ppo_training import (
    encode_grid_observation,
    extract_mona_subepisodes,
)


def test_encode_grid_observation_preserves_spatial_layout():
  flat_obs = np.array([2, 0, 1, 2, 0], dtype=np.int64)
  encoded = encode_grid_observation(
      flat_obs,
      board_shape=(2, 2),
      episode_step_limit=4,
  )
  assert encoded.shape == (4, 2, 2)
  np.testing.assert_array_equal(
      encoded[0],
      np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
  )
  np.testing.assert_array_equal(
      encoded[1],
      np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
  )
  np.testing.assert_array_equal(
      encoded[2],
      np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
  )
  np.testing.assert_allclose(encoded[3], np.full((2, 2), 0.5, dtype=np.float32))


def test_mona_subepisodes_preserve_temporal_reward_alignment():
  rollout_buffer = RolloutBuffer(
      buffer_size=3,
      observation_space=spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
      action_space=spaces.Discrete(2),
      device="cpu",
      gae_lambda=1.0,
      gamma=1.0,
      n_envs=1,
  )
  rollout_buffer.reset()

  for timestep in range(3):
    rollout_buffer.add(
        obs=np.array([[float(timestep)]], dtype=np.float32),
        action=np.array([[timestep % 2]], dtype=np.int64),
        reward=np.array([10.0 + timestep], dtype=np.float32),
        episode_start=np.array([timestep == 0], dtype=bool),
        value=torch.tensor([0.0]),
        log_prob=torch.tensor([0.0]),
    )

  subepisodes = extract_mona_subepisodes(
      rollout_buffer=rollout_buffer,
      final_observations=np.array([[3.0]], dtype=np.float32),
      final_dones=np.array([True], dtype=bool),
      optimization_len=2,
  )

  assert len(subepisodes) == 2
  first, second = subepisodes

  assert float(first[0].obs[0]) == 0.0
  assert first[0].reward == 10.0
  assert float(first[0].next_obs[0]) == 1.0
  assert not first[0].done

  assert float(first[1].obs[0]) == 1.0
  assert first[1].reward == 11.0
  assert float(first[1].next_obs[0]) == 2.0
  assert first[1].done

  assert float(second[0].obs[0]) == 1.0
  assert second[0].reward == 11.0
  assert float(second[0].next_obs[0]) == 2.0
  assert not second[0].done

  assert float(second[1].obs[0]) == 2.0
  assert second[1].reward == 12.0
  assert float(second[1].next_obs[0]) == 3.0
  assert second[1].done
