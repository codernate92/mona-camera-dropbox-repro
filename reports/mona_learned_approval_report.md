# MONA Learned Approval Report

## Motivation

The public MONA results show that myopic optimization with non-myopic approval can mitigate multi-step reward hacking in Camera Dropbox. This extension asks whether that safety benefit survives when we replace exact tabular approval with learned, noisy, misspecified approval under PPO.

## Exact Hypotheses

- H1: Approval ranking quality matters more than mean offline error for preserving PPO-time safety.
- H2: Small misspecification is more damaging under PPO than in the tabular reproduction because policy optimization actively seeks approval weaknesses.
- H3: Calibrated learned overseers preserve more intended behavior than uncalibrated learned overseers trained on the same data.
- H4: Longer optimization horizons improve observed return but increase exploitability.
- H5: There is a practical phase transition where PPO with weak approval behaves much more like ordinary RL.

## Reproduction Fidelity

- Preserved the existing public value-iteration reproduction path.
- Ported the public PPO notebook logic into scripts and reusable Python modules.
- Reused the notebook's MONA rollout-buffer recomposition callback and PPO hyperparameters as the starting point.
- Did not claim a full rerun of the paper's million-step PPO study on identical compute; this report uses smaller scripted sweeps that fit the local environment.

## Implementation Details

- `mona/src/`: copied public Camera Dropbox environment and tabular training code.
- `approval_spectrum/ppo_training.py`: scripted PPO training, MONA callback, periodic evaluation snapshots.
- `approval_spectrum/overseers.py`: oracle, noisy, misspecified, learned, and calibrated approval mechanisms.
- Learned overseers are trajectory-trained outcome models over `(time, state, action)` tuples.
- The calibration-aware variant uses explicit probability calibration (`sigmoid` or `isotonic`) before constructing the approval score.

## Experiment Matrix

- Environments: `public_camera_dropbox`, `harder_camera_dropbox`
- Approval methods: `ordinary_rl`, `oracle_mona`, `noisy_oracle_mona`, `misspecified_oracle_mona`, `learned_outcome_classifier`, `calibrated_outcome_classifier`
- Horizons: `None`, `1`, `4`
- Dataset sizes: `512`, `2048`
- Calibration methods: `none`, `sigmoid`, `isotonic`
- PPO budgets: `768`, `1536`, `3072` in the executed local pilot, with larger values supported by the scripted pipeline

## Results

- Safest public-environment run: `noisy_oracle_public_h1_budget_1536` with reward-hacking rate `0.000` and observed return `-0.366`.
- Most capable public-environment run: `noisy_oracle_public_h1_budget_1536` with observed return `-0.366` and reward-hacking rate `0.000`.
- Best learned-overseer run by true return: `calibrated_classifier_public_h1_sigmoid_ds512_budget_1536` with true return `-0.372`, intended-behavior rate `0.107`, and hacking rate `0.000`.
- In the reduced local pilot, PPO did not yet reach the strong reward-hacking regime; the dominant failure mode was slow learning and persistent failure to complete the task.
- Released public PPO MONA final behavior: fail `0.001`, intended `0.999`, hack `0.000`.
- Released public PPO ordinary RL final behavior: fail `0.007`, intended `0.077`, hack `0.915`.

## Failure Cases

- In the local pilot, the main failure mode was under-optimization: many runs still failed to complete the task reliably at these CPU-feasible budgets.
- The released public PPO reference still shows the paper's core contrast: ordinary RL converges to heavy reward hacking while MONA does not.
- Uncalibrated learned overseers can look strong offline while still having worse ranking correlation and weaker safety/capability tradeoffs than calibrated variants.

## Limitations

- PPO sweeps here are reduced-budget local runs, not identical-scale replicas of the paper's longest notebook experiments.
- Learned overseers are trained on trajectory-sampled state/action tuples rather than richer language-like oversight data.
- The calibration-aware models test one practical intervention class, not the full space of learned-approval architectures.

## Plot Artifacts

- `experiments\outputs\learned_approval\plots\safety_capability_frontier.png`
- `experiments\outputs\learned_approval\plots\training_curves.png`
- `experiments\outputs\learned_approval\plots\approval_quality.png`
- `experiments\outputs\learned_approval\plots\public_ppo_reference.png`

## Next Steps

- Push the scripted PPO runs to larger budgets and multiple seeds.
- Add a direct learned action-value overseer to compare against the outcome-model overseers here.
- Test adversarial data collection loops where PPO actively seeks states that maximize approval-model uncertainty or overestimation.
