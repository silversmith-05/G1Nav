#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skill
source /home/unitree/chenzhihong/Go2Nav/G1Nav2D/devel/setup.bash
roslaunch tool local2map_odom.launch >/tmp/local2map_odom.log 2>&1 &
sleep 2
conda deactivate

python3 /home/unitree/chenzhihong/Go2Nav/skill/realsense_publisher.py &
sleep 2
conda activate skill

python3 /home/unitree/chenzhihong/Go2Nav/skill/skill.py
