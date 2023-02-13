# rlgen
Conditional Image Generation with Reinforcement Learning

# IDEA
Abstracing the idea of diffusion models into a reinforcement learning process. Kind of inspired by [this](https://arxiv.org/abs/1805.00909). The final product probably won't be *too* different from a diffusion model, but the change in perspective will hopefully inspire different paths of progress.

# GOAL
This is a work in progress.
I am trying to get the following things working:
* Learn an image generation policy
  * Will probably initially do this with [REINFORCE](https://link.springer.com/article/10.1023/A:1022672621406) (maybe with a baseline) or a simple actor-critic algorithm
  * Goal is to eventually do this through policy iteration (similar to [MPO](https://arxiv.org/abs/1806.06920))
  * Could also try using a [decision transformer](https://arxiv.org/abs/2106.01345)
* Using an actor-critic algorithm
  * Actor (maybe should be called the artist in this case) will remove noise from the initial noisy image as its action
  * Critic will tell the actor how good the current image is, and whether to terminate an episode (i.e. stop removing noise)
* Using a decision transformer
  * Probably similar to [diffusion transformers](https://arxiv.org/abs/2212.09748)
* Environment is the forward diffusion process used to turn an initial image into random noise
  * Critic learns to predict how much noise has been added to an image (reward is higher for being closer to the original image)
  * This is a Markov chain; could be some way to extend this to be a Markov decision process (e.g. choose from multiple denoisers like in [eDiff-I](https://arxiv.org/abs/2211.01324))
  * Placed into a replay buffer like in [MPO](https://arxiv.org/abs/1806.06920) and other algorithms
  * The actor will be taking actions as a reverse diffuser, so state and next state should also be reversed
* How to calculate the reward
  * Reward ranges between 0 (noised image) and 1 (original image)
  * I am currently thinking to interpolate between the two (or more?) nearest points in the forward diffusion process to the current noise in the actor-created image
    * What should reward be if interpolation can't be performed? -1?

# RELATED STUFF
[Using actor-critic algorithms for image instance segmentation](https://arxiv.org/abs/1904.05126) \
[Using diffusion models for robust exploration in reinforcement learning](https://www.deepmind.com/publications/blade-robust-exploration-via-diffusion-models)
