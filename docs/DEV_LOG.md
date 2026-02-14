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