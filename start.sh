#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skill
source /home/unitree/chenzhihong/Go2Nav/G1Nav2D/devel/setup.bash
roslaunch tool local2map_odom.launch &
sleep 5
python3 /home/unitree/chenzhihong/Go2Nav/skill/skill.py