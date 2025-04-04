#!/usr/bin/env bash

source ~/new_pytorch_env/bin/activate

python ~/ws_robot/build/RL_robot/dqn_env/run_dqn_agent.py "$@"
