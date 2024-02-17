# **Multi Agent Reinforcement Learning Scheduling using DDPG+Woltpertinger Algorithm**


## OVERVIEW

Using DDPG algorithm with woltpertinger architecture for large discrete action space. The base code has implementation for two mobile sub-networks. The parameters are chosen 
to ensure faster convergence. Hyper parameter optimization is required when the sub networks are increased. 

## STRUCTURE

_non_stationary_channel.py_ will generate a typical factory indoor environment with two sub networks, each having 4 devices. The small scale fading coefficients using Jakes channel model and 
path loss experienced data is saved. 

_Env.py_ contains the environment class

_model2_LSTM.py_ contains the actor, critic, state processing networks

_Main.py_ contains the training script invoking the environment and actor, critic, SPN networks. 

### LICENSE

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
