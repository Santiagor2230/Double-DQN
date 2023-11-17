# Double-DQN
Implementation of Double DQN

# Requirements
gym == 0.17.3

pytorch-lightining == 1.6.0

pyglet == 1.5.27

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install pygame gym==0.17.3 pytorch-lightning==1.6.0 pyvirtualdisplay optuna

!pip install git+https://github.com/GrupoTuring/PyGame-Learning-Environment
!pip install git+https://github.com/lusob/gym-ple

!apt install swig

# Description
Double DQN is a deep Q neural model created through the works of regular DQN in order for the Deep Q Network to have a copy of its own network called a target network, this implementation allows us to converge slowly to an optimal solution without having the network generalizing to noise and having a target network with its own copy of the original network allows us to optimize slowly. We take into consideration the help of both the value function and the advantage function in order to have a more stable q network with higher prediction on state and action pairs. Rewards are used to compared and find the loss of the DQN in order to get to the most optimal path in the game that will maximize the rewards that the agent can obtain.

# Game
Flappy Bird

# Architecture
Double DQN

# optimizer
AdamW

# loss function
smooth L1 loss function

# Video Results:
https://github.com/Santiagor2230/Double-DQN/assets/52907423/8746e571-c498-4f83-a667-dc04ffbd0bbc

