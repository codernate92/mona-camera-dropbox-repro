"""Configuration objects and curated experiment suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
  """Camera Dropbox environment parameters."""

  name: str
  board_shape: tuple[int, int]
  max_boxes: int
  min_blocking_boxes: int = 1
  per_step_penalty: float = 0.01
  episode_step_limit: int = 50

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class PPOConfig:
  """PPO training parameters."""

  total_timesteps: int = 10_000
  save_interval: int = 2_000
  n_steps: int = 256
  batch_size: int = 64
  ent_coef: float = 0.05
  clip_range: float = 0.3
  learning_rate: float = 5e-5
  gamma: float = 1.0
  device: str = "auto"
  policy: str = "MlpPolicy"

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
  """Trajectory dataset parameters for learned overseers."""

  num_samples: int = 2048
  behavior_policy_mix: tuple[str, ...] = ("random", "good", "bad")
  hidden_layer_sizes: tuple[int, ...] = (64, 64)
  max_iter: int = 400

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class ApprovalConfig:
  """Approval construction parameters."""

  method: str
  horizon: int | None = None
  noise_scale: float = 0.0
  misspecification: float = 0.0
  calibration_method: str = "none"
  dataset: DatasetConfig | None = None

  def to_dict(self) -> dict[str, object]:
    data = asdict(self)
    if self.dataset is None:
      data["dataset"] = None
    return data


@dataclass(frozen=True)
class ExperimentSpec:
  """A single PPO training/evaluation run."""

  name: str
  env_name: str
  approval: ApprovalConfig
  ppo: PPOConfig
  seed: int = 0

  def to_dict(self) -> dict[str, object]:
    return {
        "name": self.name,
        "env_name": self.env_name,
        "approval": self.approval.to_dict(),
        "ppo": self.ppo.to_dict(),
        "seed": self.seed,
    }


ENVIRONMENT_PRESETS: dict[str, EnvironmentConfig] = {
    "public_camera_dropbox": EnvironmentConfig(
        name="public_camera_dropbox",
        board_shape=(4, 4),
        max_boxes=2,
    ),
    "harder_camera_dropbox": EnvironmentConfig(
        name="harder_camera_dropbox",
        board_shape=(4, 4),
        max_boxes=3,
        min_blocking_boxes=2,
        episode_step_limit=60,
    ),
}


DEFAULT_PPO_CONFIG = PPOConfig(
    total_timesteps=1_536,
    save_interval=512,
    n_steps=128,
    batch_size=64,
)
FAST_PPO_CONFIG = PPOConfig(
    total_timesteps=768,
    save_interval=256,
    n_steps=128,
    batch_size=64,
)
LONGER_PPO_CONFIG = PPOConfig(
    total_timesteps=3_072,
    save_interval=1_024,
    n_steps=128,
    batch_size=64,
)


def build_report_suite(seed: int = 0) -> list[ExperimentSpec]:
  """Curated suite small enough to run locally but broad enough to be meaningful."""
  learned_small = DatasetConfig(num_samples=512)
  learned_large = DatasetConfig(num_samples=2048)
  specs = [
      ExperimentSpec(
          name="ordinary_rl_public_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(method="ordinary_rl"),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="oracle_mona_public_h1_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(method="oracle_mona", horizon=1),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="oracle_mona_public_h4_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(method="oracle_mona", horizon=4),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="noisy_oracle_public_h1_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="noisy_oracle_mona",
              horizon=1,
              noise_scale=0.25,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="misspecified_oracle_public_h1_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="misspecified_oracle_mona",
              horizon=1,
              misspecification=0.25,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="learned_classifier_public_h1_ds512_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="learned_outcome_classifier",
              horizon=1,
              dataset=learned_small,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="learned_classifier_public_h1_ds2048_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="learned_outcome_classifier",
              horizon=1,
              dataset=learned_large,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="calibrated_classifier_public_h1_sigmoid_ds512_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="calibrated_outcome_classifier",
              horizon=1,
              calibration_method="sigmoid",
              dataset=learned_small,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="calibrated_classifier_public_h1_isotonic_ds2048_budget_1536",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(
              method="calibrated_outcome_classifier",
              horizon=1,
              calibration_method="isotonic",
              dataset=learned_large,
          ),
          ppo=DEFAULT_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="oracle_mona_harder_h1_budget_768",
          env_name="harder_camera_dropbox",
          approval=ApprovalConfig(method="oracle_mona", horizon=1),
          ppo=FAST_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="ordinary_rl_harder_budget_768",
          env_name="harder_camera_dropbox",
          approval=ApprovalConfig(method="ordinary_rl"),
          ppo=FAST_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="calibrated_classifier_harder_h1_isotonic_ds2048_budget_768",
          env_name="harder_camera_dropbox",
          approval=ApprovalConfig(
              method="calibrated_outcome_classifier",
              horizon=1,
              calibration_method="isotonic",
              dataset=learned_large,
          ),
          ppo=FAST_PPO_CONFIG,
          seed=seed,
      ),
  ]
  return specs


def build_scripted_ppo_reproduction_suite(seed: int = 0) -> list[ExperimentSpec]:
  """Minimal scripted PPO reproduction suite derived from the notebook."""
  return [
      ExperimentSpec(
          name="ppo_public_ordinary_rl",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(method="ordinary_rl"),
          ppo=LONGER_PPO_CONFIG,
          seed=seed,
      ),
      ExperimentSpec(
          name="ppo_public_mona_oracle_h1",
          env_name="public_camera_dropbox",
          approval=ApprovalConfig(method="oracle_mona", horizon=1),
          ppo=LONGER_PPO_CONFIG,
          seed=seed,
      ),
  ]
