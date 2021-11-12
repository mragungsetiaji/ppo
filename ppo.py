import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from typing import Tuple, List

class PPO:
    
    def __init__(self, policy_class, env, **hyperparameters) -> None:
        """Initializes the PPO model, include with hyperparameters.

        Args:
            policy_class ([type]): the policy class to use for our actor/critic networks.
            env ([type]): the environment to train on.
            hyperparameters: all extra arguments passed into PPO
        """
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
            "delta_t":time.time_ns(),
            "timestep":0,
            "iteration":0,
            "batch_lengths":[],
            "batch_rewards":[],
            "actor_losses":[],
        }

    def _init_hyperparameters(self, hyperparameters: dict) -> None:
        """Initializes the hyperparameters for the PPO model.

        Args:
            hyperparameters (dict): all extra arguments passed into PPO
        """
        # Initialize hyperparameters with default values
        self.timesteps_per_batch = 4800 # Number of timesteps to run per batch
        self.max_timestamps_per_episode = 1600 # Maximum number of timesteps per episode
        self.n_updates_per_iteration = 5 # Number of times to update actor/critic per iteration
        self.learning_rate = hyperparameters.get('learning_rate', 0.005) # Learning rate for actor/critic optimizer
        self.gamma = hyperparameters.get('gamma', 0.95) # Discount factor to be used for future rewards
        self.clip_ratio = hyperparameters.get('clip_ratio', 0.2) # Clipping ratio for the surrogate objective
        
        # Other params
        self.render = hyperparameters.get('render', True) # Whether to render the environment or not
        self.render_every_i = hyperparameters.get('render_every_i', 10) # How often to render the environment
        self.save_freq = hyperparameters.get('save_freq', 10) # How often to save the model
        self.seed = hyperparameters.get('seed', 0) # Random seed, used for reproducibility of results

        # Change any default valus to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec(f"self.{param} = {val}")

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)

    def get_action(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an action from the actor network, called from rollout().

        Args:
            observation (torch.Tensor): the observation to get an action for. 
                at current timestep.

        Returns:
            action (torch.Tensor): the action to take.
            log_prob (torch.Tensor): the log probability of the action taken.
        """
        # Query the actor network for a mean action.
        mean = self.actor(observation) 

        # Create a distribution with mean action and stf from the covariance
        # matrix above. Reference: https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution.
        action = dist.sample()

        # Calculate the log probability of the action taken.
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rewards: torch.Tensor) -> torch.Tensor:
        """Computes the Reward-To-Go of each timestep in a batch given the rewards.

        Args:
            batch_rewards (torch.Tensor): the rewards for each timestep in the batch.

        Returns:
            batch_rtgs (torch.Tensor): the rewards to go.
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (number timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0 # The discounted reward for this episode

            # Iterate through all rewards in the episode. We go backwards for
            # smoother calculation of each discounted return (think about why
            # it would be harder starting from the beginning of the episode).
            for reward in reversed(episode_rewards):
                # Discount the reward by the gamma factor
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs).float()
        return batch_rtgs

    def rollout(self) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
        """ Rollout the environment for a batch of episodes. Collect the batch of 
            data from simulation. Since this is an on-policu algorithm, we'll 
            need to collect a fresh batch of data each time we iterate 
            the actor/critic networks.

        Returns:
            batch_obervations (torch.Tensor): the observations collected 
                this batch. Shape: (number of timesteps, dimension of observation)
			batch_actions (torch.Tensor): the actions collected this batch. 
                Shape: (number of timesteps, dimension of action)
			batch_log_probs (torch.Tensor): the log probabilities of each action 
                taken this batch. Shape: (number of timesteps)
			batch_rtgs (torch.Tensor): the Rewards-To-Go of each timestep in 
                this batch. Shape: (number of timesteps)
			batch_lengths (torch.Tensor): the lengths of each episode this batch. 
                Shape: (number of episodes)
        """
        # Batch data. 
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lengths = [] 

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode.
        episode_rewards = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch.

        # Keep simulating until we've run more than or equal to specified timesteps
        # per batch.
        while t < self.timesteps_per_batch:
            episode_rewards = [] # Rewards collected per episode.

            # Reset the environment. Note that obs is short for observation.
            observation = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps per episode.
            for episode_t in range(self.max_timesteps_per_episode):
                # Render the environment if specified
                if self.render and episode_t % self.render_every_i == 0:
                    self.env.render()
                
                t += 1 # Increment the number of timesteps we've run so far this batch.

                # Track observations in this batch
                batch_observations.append(observation)

                # Calculate action and make a step in the env.
                action, log_prob = self.get_action(torch.tensor(observation).float())
                observation, reward, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                # If we're done, break out of the episode loop.
                if done:
                    break

            # Track episodic lenghts and rewards
            batch_lengths.append(episode_t + 1)
            batch_rewards.append(episode_rewards)

        # Reshape data as tensors in the shape specified in function description,
        # before returning.
        batch_observations = torch.tensor(batch_observations).float()
        batch_actions = torch.tensor(batch_actions).float()
        batch_log_probs = torch.tensor(batch_log_probs).float()
        batch_rtgs = self.compute_rtgs(batch_rewards)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rewards"] = batch_rewards
        self.logger["batch_lengths"] = batch_lengths

        return (
            batch_observations, batch_actions, batch_log_probs, batch_rtgs, 
            batch_lengths
        )
    
    def evaluate(self, 
            batch_observations: torch.Tensor, 
            batch_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Estimate the values of each observation, and log probs of each
            action in the most recent batch with the most recent iteration
            of the actor/critic networks. Called from learn().

        Args:
            batch_observations (torch.Tensor): the observations collected this 
                batch. Shape: (number of timesteps, dimension of observation)
            batch_actions (torch.Tensor): the actions collected this batch. 
                Shape: (number of timesteps, dimension of action)

        Returns:
            V (torch.Tensor): the estimated values of batch_observations.
            log_probs (torch.Tensor): the log probabilities of batch_actions.
        """
        # Query critic network for a value V for each bacth_observations.
        # Shape of V should be same as batch_rtgs.
        V = self.critic(batch_observations).squeeze()

        # Calculate the log probabilities of batch_actions using most recent
        # actor network. This segment of code is similar to that in get_action()
        mean = self.actor(batch_observations)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)

        # Return the value vector V of each observation in the batch and log
        # probability log_probs of each action in the batch.
        return V, log_probs

    def _log_summary(self) -> None:
        """ Print to stdout what we've logged so far in the most recent batch.
        """
        # Calculate logging values.
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        timestep = self.logger['timestep']
        iteration = self.logger['iteration']
        avg_ep_lens = np.mean(self.logger['batch_lengths'])
        avg_ep_rews = np.mean([
            np.sum(ep_rews) for ep_rews in self.logger['batch_rewards']])
        avg_actor_loss = np.mean([
            losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{iteration} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {timestep}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lengths'] = []
        self.logger['batch_rewards'] = []
        self.logger['actor_losses'] = []

    def learn(self, total_timesteps:int) -> None:
        """Train the actor and critic networks. Here is where the main
           PPO algoritm resides.

        Args:
            total_timesteps (int): the total number of timestamps to train for
        """
        print(f"Learning, Running {self.max_timestamps_per_episode} timestamps per episode, ", end="")
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        timestep = 0
        iteration = 0
        while timestep < total_timesteps:
            # Collecting batch simulations
            (
                batch_observations, batch_actions, batch_log_probs, batch_rtgs, 
                batch_lenghts
            ) = self.rollout()

            # Calculate how many timesteps we collected this batch
            timestep += np.sum(batch_lenghts)

            # Increment the number of iterations
            iteration += 1

            # Log the number of timesteps we've run so far
            self.logger["timestep"] = timestep
            self.logger["iteration"] = iteration

            # Calculate the advantage function at k-th iteration
            V, _ = self.evaluate(batch_observations, batch_actions)
            A_k = batch_rtgs - V.detach()

            # One of the only tricks I use that isn't in the pseudocode.
            # Normalizing advantages isnt theoritically necessary, but in practice
            # it decreases the variance of our advantages and makes convergence
            # much more stable and faster. I added this because solving 
            # some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs.
            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t|s_t) 
                V, curr_log_probs = self.evaluate(batch_observations, batch_actions)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because 
                # we're trying to maximize the performance function, but Adam 
                # minimizes the loss. So minimizing the negative performance 
                # function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for 
                # critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
            
            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if iteration % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')