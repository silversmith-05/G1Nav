#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skill
source /home/unitree/G1Nav/G1Nav2D/devel/setup.bash
roslaunch tool local2map_odom.launch &
python /home/unitree/G1Nav/skill/skill.py