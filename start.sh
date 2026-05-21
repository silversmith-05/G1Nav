#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skill
source /home/unitree/G1Nav/G1Nav2D/devel/setup.bash
roslaunch tool local2map_odom.launch >/tmp/local2map_odom.log 2>&1 &
sleep 2
conda deactivate

python3 /home/unitree/G1Nav/skill/gemini_publisher.py &
sleep 2
conda activate skill

python /home/unitree/G1Nav/skill/skill.py
