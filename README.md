# Transfer-Learning-In-RL
design a reinforcement learning algorithm that leverages prior experience to figure out how to solve new tasks quickly

# How to run
## Supported games:
* CartPole-v1 - `cart`
* MountainCarContinuous-v0 - `car`
* Acrobot-v1 - `acro`

## Train single model
`--mode=train --trained_model=cart --test_game=cart`

## Fine tune model for new game
`--mode=fine --trained_model=acro --test_game=cart`

## Transfer learning using progressive network from 2 games to a new game
`--mode=transfer --trained_model=cart --trained_model2=acro --test_game=car`