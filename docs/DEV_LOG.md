# Development Log

## Init Protocol Record
- **Date**: 2026-02-14
- **Action**: Global Formatting Completed
- **Environment**:
    - Python: 3.10.12
    - Conda Env: None (System Python)

## Project Analysis & Documentation (2026-02-14)

### Modifications
- **Created `docs/PROJECT_OVERVIEW.md`**: Generated a comprehensive project overview document as per `AGENT.md` requirements. This document details the project structure, core modules (Engines, Envs, Agents, Anim), data flow, and key algorithms.

### Key Commands
- Analyzed project structure using `ls`, `grep` and file reading tools.
- Installed Fish Shell: `sudo apt-get install -y fish`.
- Initialized Conda for Fish Shell: `/home/hz/miniconda3/bin/conda init fish`.

### Troubleshooting
- **Issue**: `CondaError: Run 'conda init' before 'conda activate'`.
- **Cause**: The current shell session has not initialized Conda's shell functions, so `conda activate` cannot modify the environment variables.
- **Solution**: Run `source ~/miniconda3/etc/profile.d/conda.sh` to enable activation in the current session, or `conda init zsh` for permanent setup.

- **Issue**: `CondaError` in Fish Shell.
- **Cause**: Fish Shell requires its own Conda initialization, separate from Bash/Zsh.
- **Solution**: Run `source ~/miniconda3/etc/fish/conf.d/conda.fish` for the current session, then `conda init fish` for permanent setup.



---
test
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_lab_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml --visualize true --out_dir output/

---

## Isaac Lab Engine Fix & G1 AMP Training (2026-02-14)

### Modifications
- **Fixed `IsaacLabEngine` AttributeError**: Resolved `AttributeError: 'Articulation' object has no attribute 'has_external_wrench'`.
    - **Cause**: The engine tried to clear external forces by checking a `has_external_wrench` attribute that wasn't initialized on the `Articulation` or `RigidObject` instances.
    - **Fix**: Initialized the attribute in `_build_objs`, updated it in `set_body_forces`, and added safety logic in `_clear_forces`.
- **Verified Humanoid Training**: Successfully ran the first iterations of humanoid AMP training using the fixed engine.

### G1 AMP Training (Isaac Lab)
To train the G1 robot using AMP and the Isaac Lab engine:

1. **Environment**: `conda activate mimickit`
2. **Command**:
```bash
python mimickit/run.py \
    --env_config data/envs/amp_g1_env.yaml \
    --engine_config data/engines/isaac_lab_engine.yaml \
    --agent_config data/agents/amp_g1_agent.yaml \
    --headless
```
3. **Motions**: Default is `g1_walk.pkl`. Use `--motion_file` to switch to `g1_run.pkl` or `g1_spinkick.pkl`.

### Troubleshooting Note: SIGKILL / OOM
If the training crashes with `SIGKILL` during "Initializing simulation", it is likely due to the high resource demands of the G1 model at scale. 
- **Finding**: 1024 envs (2048 robots) may exceed initialization resource limits. 16 envs have been verified to work correctly.
- **Recommendation**: Start with `--num_envs 256` or `512`. The G1 model is more complex than the standard Humanoid, so it requires fewer parallel environments to fit in VRAM/RAM during initialization.


---

python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_location_humanoid_env.yaml \
  --agent_config data/agents/amp_task_humanoid_agent.yaml \
  --visualize false \
  --out_dir output/



  python mimickit/run.py \
  --mode train \
  --num_envs 8192 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_g1_env.yaml \
  --agent_config data/agents/amp_g1_agent.yaml \
  --visualize false \
  --out_dir output/

转换脚本
  python tools/csv_to_mimickit.py --input data/csv/walk1_subject1.csv --output data/motions/g1/test.pkl               

测试脚本
data/envs/view_motion_g1_env.yaml
修改地址
python mimickit/run.py \
  --mode test \
  --num_envs 4 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/view_motion_g1_env.yaml \
  --visualize true


---
训练
python mimickit/run.py \
  --mode train \
  --num_envs 8192 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_steering_g1_env.yaml \
  --agent_config data/agents/amp_task_g1_agent.yaml \
  --visualize false \
  --out_dir output/
