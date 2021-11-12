import gym
from torch import nn
from typing import Tuple

def _log_summary(
        episode_length: float, 
        episode_return: float, 
        episode_number: int
    ):
    """ Print to stdout what we've logged so far in the most recent episode.
        
        Args:
            episode_length: The length of the most recent episode.
            episode_return: The return of the most recent episode.
            episode_number: Index of recent episode.
    """
    # Round decimal places for more aesthetic logging messages
    episode_length = str(round(episode_length, 2))
    episode_return = str(round(episode_return, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------- Episode #{episode_number} ---------------", flush=True)
    print(f"Episodic Length: {episode_length}", flush=True)
    print(f"Episodic Return: {episode_return}", flush=True)
    print(f"------------------------------------------", flush=True)
    print(flush=True)

def rollout(policy: nn.Module, env:gym.Env, render:bool=False) -> Tuple[float, float]:
    """ Returns a generator to roll out each episode given a trained policy and
        environment to test on.

        Args:
            policy : The trained policy to test
            env (gym.Env): The environment to evaluate the policy on
            render (bool): Specifies whether to render or not. Default False.
        
        Returns:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.
        Note:
            If you're unfamiliar with Python generators, check this out:
                https://wiki.python.org/moin/Generators
            If you're unfamiliar with Python "yield", check this out:
                https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """
    # Rollout until user kills process
    while True:
        observation = env.reset()
        done = False

        # Number of timesteps so far
        t = 0

        # Logging data
        episode_length = 0  # episodic length
        episode_return = 0  # episodic return

        while not done:
            t += 1

            # Render environment if specified, off by default
            if render:
                env.render()

            # Query deterministic action from policy and run it
            action = policy(observation).detach().numpy()
            observation, reward, done, _ = env.step(action)

            # Sum all episodic rewards as we go along
            episode_return += reward
            
        # Track episodic length
        episode_length = t

        # returns episodic length and return in this iteration
        yield episode_length, episode_return

def eval_policy(policy: nn.Module, env: gym.Env, render:bool=False) -> None:
    """ The main function to evaluate our policy with. It will iterate a 
        generator object "rollout", which will simulate each episode and 
        return the most recent episode's length and return. We can then log 
        it right after. And yes, eval_policy will run forever until you kill 
        the process. 
        
        Parameters:
            policy : The trained policy to test, basically another name for 
                our actor model
            env (gym.Env): The environment to test the policy on
            render (bool): Whether we should render our episodes. Default False.
    """
    # Rollout with the policy and environment, and log each episode's data
    for episode_number, (episode_length, episode_return) in enumerate(
            rollout(policy, env, render)):
        _log_summary(
            episode_length=episode_length, 
            episode_return=episode_return, 
            episode_number=episode_number)