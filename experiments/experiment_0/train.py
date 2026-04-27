import json
import argparse
import os
from pathlib import Path


import copy
import tempfile
import numpy as np

import torch

from matplotlib import pyplot as plt
from tensordict import TensorDictBase # The base class for tensordicts, which are used to store and manipulate data in a structured way.
from tensordict import TensorDict



from tensordict.nn import TensorDictModule, TensorDictSequential # These are modules that operate on tensordicts, allowing for the creation of complex neural network architectures that can handle structured data.
from torch import multiprocessing # This is used for parallel processing, allowing for the creation of multiple processes to run concurrently, which can be useful for tasks like data collection or training.

from torchrl.collectors import Collector # This is a class for collecting data synchronously from an environment, which can be used for training reinforcement learning agents.
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer # These are classes for storing and sampling data. LazyMemmapStorage allows for efficient storage of large datasets on disk, RandomSampler is used to sample data randomly from a dataset, and ReplayBuffer is a common structure used in reinforcement learning to store past experiences for training.

from torchrl.envs import (
    check_env_specs, # This function checks the specifications of an environment to ensure they are valid and compatible with the expected format.
    ExplorationType, # This is an enumeration that defines different types of exploration strategies that can be used in reinforcement learning.
    PettingZooEnv,  # This is a wrapper for environments from the PettingZoo library, which provides a collection of multi-agent environments for reinforcement learning research.
    RewardSum, # This is a class that computes the sum of rewards over time, which can be used to evaluate the performance of a reinforcement learning agent.
    set_exploration_type, # This function is used to set the exploration type for a reinforcement learning agent, which determines how the agent explores the environment during training.
    TransformedEnv, # This is a class that allows for the transformation of an environment using a sequence of transformations, which can be useful for preprocessing observations or rewards before they are used by a reinforcement learning agent.
    VmasEnv, # This is a wrapper for environments that are compatible with the Vmas (Vectorized Multi-Agent Systems) interface, which allows for efficient handling of multiple agents in a vectorized manner.
    PettingZooWrapper
)

from torchrl.modules import (
    AdditiveGaussianModule, # This is a module that adds Gaussian noise to its input, which can be used for exploration in reinforcement learning.
    MultiAgentMLP, # This is a multi-agent version of a multi-layer perceptron (MLP), which can be used to create neural networks that can handle multiple agents in a reinforcement learning setting.
    ProbabilisticActor, # This is a module that represents a probabilistic policy, which outputs a distribution over actions given an observation, and can be used for stochastic policies in reinforcement learning.
    TanhDelta, # This is a module that applies a hyperbolic tangent (tanh) transformation to its input, which can be used to ensure that the output of a neural network is bounded within a certain range, often used in reinforcement learning for action outputs.
    OrnsteinUhlenbeckProcessModule # This is a module that implements the Ornstein-Uhlenbeck process, which is a type of stochastic process that can be used to generate temporally correlated noise for exploration in reinforcement learning, particularly in continuous action spaces.
)

from torchrl.objectives import (
    DDPGLoss, # This is a class that implements the loss function for the Deep Deterministic Policy Gradient (DDPG) algorithm, which is a reinforcement learning algorithm for continuous action spaces.
    SoftUpdate, # This is a class that implements the soft update mechanism for target networks in reinforcement learning, which helps to stabilize training by slowly updating the target network parameters towards the main network parameters.
    ValueEstimators, # This is a class that provides various methods for estimating the value function in reinforcement learning, which can be used to evaluate the expected return of a given state or state-action pair.
)

from torchrl.record import (
    CSVLogger, # This is a class that allows for logging data to a CSV file, which can be useful for tracking the performance of a reinforcement learning agent over time.
)

from tqdm import tqdm # This is a library that provides a progress bar for loops, which can be useful for tracking the progress of data collection or training in reinforcement learning.

import sys 

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_path)

from network_env.network_env_v0 import parallel_env # This is a function that creates a parallel environment for the network slicing task, which allows for efficient handling of multiple agents in a multi-agent reinforcement learning setting.

def load_config(file_path):
    """Reads and parses the JSON configuration file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    try:
        with open(path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Read MARL configuration parameters from a JSON file.")
    parser.add_argument("config_path_experiment", type=str, help="Path to the JSON configuration file")
    parser.add_argument("config_path_resource", type=str, help="Path to the JSON configuration file")
    parser.add_argument("log_path", type=str, help="logging results location")
    parser.add_argument("traffic_path", type=str, help="traffic")


    
    args = parser.parse_args()
    config_exp = load_config(args.config_path_experiment)
    

    if config_exp:
        # Example of accessing the specific parameters you defined
        print(f"--- Configuration Loaded: {args.config_path_experiment} ---")
        print(f"Agents: {config_exp.get('n_agent')}")
        print(f"MECs: {config_exp.get('n_mecs')}")
        print(f"Learning Rate: {config_exp.get('lr')}")
        
        # Accessing nested dictionaries
        critic_cells = config_exp.get("critic", {}).get("num_cells")
        policy_cells = config_exp.get("policy", {}).get("num_cells")
        
        print(f"Critic Hidden Units: {critic_cells}")
        print(f"Policy Hidden Units: {policy_cells}")
        print(f"Centralized Critic: {config_exp.get('critic', {}).get('centralized_critic')}")
        print("------------------------------------------")
        
        # You can now pass 'config_exp' into your Environment or Model builders
        #return config_exp
    
    # read config
    n_agent = config_exp.get('n_agent')
    frames_per_batch = config_exp.get('transition_per_batch')
    n_iters = config_exp.get('n_iterations')
    min_replay_size = config_exp.get('min_replay_size')
    memory_size = config_exp.get('memory_size')
    n_optimizer_steps = config_exp.get('n_optimizer_steps')
    train_batch_size = config_exp.get('training_batch_size')
    lr = config_exp.get('lr')
    gamma = config_exp.get('gamma')
    polyak_tau = config_exp.get('polyak_tau')
    max_grad_norm = config_exp.get('max_grad_norm')
    total_frames = frames_per_batch * n_iters
    
    critic_num_cells = config_exp.get("critic").get('num_cells')
    critic_depth = config_exp.get('critic').get('depth')
    critic_share_parameter = config_exp.get('critic').get('share_parameter')
    critic_centralized_critic = config_exp.get('critic').get('centralized_critic')
    
    policy_num_cells = config_exp.get("policy").get('num_cells')
    policy_depth = config_exp.get('policy').get('depth')
    policy_share_parameter = config_exp.get('policy').get('share_parameter')
    
    noise_sigma_init = config_exp.get('noise').get('sigma_init')
    noise_sigma_end = config_exp.get('noise').get('sigma_end')
    
    # Seed
    seed = 0
    torch.manual_seed(seed)

    # Devices
    is_fork = multiprocessing.get_start_method() == "fork"
    #device = (
    #    torch.device(0)
    #    if torch.cuda.is_available() and not is_fork
    #    else torch.device("cpu")
    #)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    
    # creat env
    print(args.config_path_resource)
    custom_env = parallel_env(config_path=args.config_path_resource, num_agents=1, traffic_path=args.traffic_path, log_path=args.log_path)
    
    env = PettingZooWrapper(
    env=custom_env,
    use_mask=True,                 # Required if agents can "die" or for AEC envs
    group_map=None,                # Defaults to one group per agent name
    #group_map= None,
    categorical_actions=False,       # Useful if your actions are Discrete
    )
    
    #transformed_env = TransformedEnv(
    #    env,
    #    RewardSum(
    #        in_keys=env.reward_keys,
    #        reset_keys=["_reset"] * len(env.group_map.keys()),
    #    ),
    #)
    
    
    #create policy
    policy_modules = {}
    for group, agents in env.group_map.items():
        share_parameters_policy = policy_share_parameter  # Can change this based on the group

        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=env.full_action_spec[group, "action"].shape[
                -1
            ],  # n_actions_per_agents
            n_agents=len(agents),  # Number of agents in the group
            centralised=False,  # the policies are decentralised (i.e., each agent will act from its local observation)
            share_params=share_parameters_policy,
            device=device,
            depth=policy_depth,
            num_cells=policy_num_cells,
            activation_class=torch.nn.Tanh,
        )

        # Wrap the neural network in a :class:`~tensordict.nn.TensorDictModule`.
        # This is simply a module that will read the ``in_keys`` from a tensordict, feed them to the
        # neural networks, and write the
        # outputs in-place at the ``out_keys``.

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "param")],
        )  # We just name the input and output that the network will read and write to the input tensordict
        policy_modules[group] = policy_module
        
    policies = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": env.full_action_spec_unbatched[group, "action"].space.low,
                "high": env.full_action_spec_unbatched[group, "action"].space.high,
            },
            return_log_prob=False,
        )
        policies[group] = policy
    
    #create noise
    exploration_policies = {}
    for group, _agents in env.group_map.items():
        exploration_policy = TensorDictSequential(
            policies[group],
            AdditiveGaussianModule(
                spec=policies[group].spec,
                annealing_num_steps=total_frames
                // 2,  # Number of frames after which sigma is sigma_end
                action_key=(group, "action"),
                sigma_init=noise_sigma_init,  # Initial value of the sigma
                sigma_end=noise_sigma_end,  # Final value of the sigma
            ),
        )
        exploration_policies[group] = exploration_policy
    
    
    #create critic
    critics = {}
    for group, agents in env.group_map.items():
        share_parameters_critic = critic_share_parameter  # Can change for each group
        MADDPG = critic_centralized_critic  # IDDPG if False, can change for each group

        # This module applies the lambda function: reading the action and observation entries for the group
        # and concatenating them in a new ``(group, "obs_action")`` entry
        cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[(group, "observation"), (group, "action")],
            out_keys=[(group, "obs_action")],
        )

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
                + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs=1,  # 1 value per agent
                n_agents=len(agents),
                centralised=MADDPG,
                share_params=share_parameters_critic,
                device=device,
                depth=critic_depth,
                num_cells=critic_num_cells,
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[(group, "obs_action")],  # Read ``(group, "obs_action")``
            out_keys=[
                (group, "state_action_value")
            ],  # Write ``(group, "state_action_value")``
        )

        critics[group] = TensorDictSequential(
            cat_module, critic_module
        )  # Run them in sequence
    
    # create data collector
    # Put exploration policies from each group in a sequence
    agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

    collector = Collector(
        env,
        agents_exploration_policy,
        device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    
    #create replay buffer
    replay_buffers = {}
    scratch_dirs = []
    for group, _agents in env.group_map.items():
        scratch_dir = tempfile.TemporaryDirectory().name
        scratch_dirs.append(scratch_dir)
        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(
                memory_size,
                scratch_dir=scratch_dir,
            ),  # We will store up to memory_size multi-agent transitions
            sampler=RandomSampler(),
            batch_size=train_batch_size,  # We will sample batches of this size
        )
        if device.type != "cpu":
            replay_buffer.append_transform(lambda x: x.to(device))
        replay_buffers[group] = replay_buffer
    
    #creat loss function
    losses = {}
    for group, _agents in env.group_map.items():
        loss_module = DDPGLoss(
            actor_network=policies[group],  # Use the non-explorative policies
            value_network=critics[group],
            delay_value=True,  # Whether to use a target network for the value
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

        losses[group] = loss_module

    target_updaters = {
        group: SoftUpdate(loss, tau=polyak_tau) for group, loss in losses.items()
    }

    optimisers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(), lr=lr
            ),
            "loss_value": torch.optim.Adam(
                loss.value_network_params.flatten_keys().values(), lr=lr
            ),
        }
        for group, loss in losses.items()
    }
    
    
    def process_batch(batch: TensorDictBase) -> TensorDictBase:
        """
        If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
        `"terminated"` and `"done"`.
        This is needed to present them with the same shape as the reward to the loss.
        """
        for group in env.group_map.keys():
            keys = list(batch.keys(True, True))
            group_shape = batch.get_item_shape(group)
            nested_done_key = ("next", group, "done")
            nested_terminated_key = ("next", group, "terminated")
            if nested_done_key not in keys:
                batch.set(
                    nested_done_key,
                    batch.get(("next", "done")).unsqueeze(-1).expand(1, group_shape[0]),
                )
            if nested_terminated_key not in keys:
                batch.set(
                    nested_terminated_key,
                    batch.get(("next", "terminated")).unsqueeze(-1).expand(1, group_shape[0]),
                )
        return batch
    
    
    # --- Setup Constants ---


    # 1. Initialize logic
    pbar = tqdm(total=n_iters)
    reward_history_map = {group: [] for group in env.group_map.keys()}
    train_group_map = copy.deepcopy(env.group_map)
    iteration_rewards = {group: [] for group in env.group_map.keys()}

    incomplete_transitions = {group: {} for group in env.group_map.keys()}  # {group: {ticket_id: (time_step, transition)}}

    synchronized_timer = 0

    for iteration, batch in enumerate(collector):
        current_frames = batch.numel()    
        # Pre-process batch (handling masking, global state, etc.)
        batch = process_batch(batch) 
        #print(batch)
        #print(batch["next"]["agent"]["reward"])
        #print(batch[0])
        
        # The batch size of the collector output corresponds to the steps taken
        for idx, time in enumerate(range(synchronized_timer, synchronized_timer + frames_per_batch)):
            single_step_td = batch[idx]
            
            for group in train_group_map.keys():
                group_data = single_step_td.exclude(
                    *[key for _group in env.group_map.keys() if _group != group 
                        for key in [_group, ("next", _group)]]
                )
                
                #print(f'group data \n {group_data}')
                
                incomplete_transitions[group][time] = group_data.clone().reshape(-1)
                
            
        for group in train_group_map.keys():

                
            if env.is_ready():
                ready_reward = env.get_ready_reward()
                
                #print(f'UPDATE INCOMPLETE TRANSITION {list(ready_reward.keys())}')
                #print(ready_reward)
                
                for time, reward in sorted(ready_reward.items()):
                    #print(f'Time step, Reward {time,reward}')
                    #print(f"old reward: {incomplete_transitions[group][time]['next']['agent']['reward']}")
                    
                    #print(complete_transition['next',group,'reward'].shape)
                    #print(complete_transition['next', group, 'done'].shape)
                    
                    #incomplete_transitions[group][time]['next']['agent']['reward'] = torch.tensor(list(reward.values())).reshape([1,4])
                    complete_transition = incomplete_transitions[group].pop(time)
                    reward_tensor = torch.tensor(np.array(list(reward.values())).reshape(1, n_agent, 1), dtype=torch.float32)
                    complete_transition.set(("next", group, "reward"), reward_tensor)
                    
                
                    #print(f"new reward: {complete_transition['next'][group]['reward']}")
                    
                    
                    #incomplete_transitions[group][time].reshape(-1)
                    replay_buffers[group].extend(complete_transition)
                    
                    iteration_rewards[group].append(reward_tensor.mean().item())

                    #print(incomplete_transitions)
                    
        synchronized_timer += frames_per_batch

        # --- OPTIMIZATION BLOCK ---
        for group in train_group_map.keys():
            # Only train if we have enough "realized" transitions
            if len(replay_buffers[group]) >= min_replay_size:
                for _ in range(n_optimizer_steps):
                    subdata = replay_buffers[group].sample()
                    loss_vals = losses[group](subdata)

                    for loss_name in ["loss_actor", "loss_value"]:
                        loss = loss_vals[loss_name]
                        optimiser = optimisers[group][loss_name]
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            optimiser.param_groups[0]["params"], 
                            max_grad_norm
                        )
                        optimiser.step()
                        optimiser.zero_grad()

                    target_updaters[group].step()
                
                # Decay exploration if applicable
                exploration_policies[group][-1].step(current_frames)

    # --- LOGGING ---
        current_log_status = []
        for group in env.group_map.keys():
            # Move iteration rewards to the permanent history map
            if iteration_rewards[group]:
                avg_reward_this_step = sum(iteration_rewards[group]) / len(iteration_rewards[group])
                reward_history_map[group].append(avg_reward_this_step)
                iteration_rewards[group] = [] # Reset for next batch
            
            # Calculate rolling mean for a stable progress bar
            history = reward_history_map[group]
            rolling_mean = sum(history[-20:]) / len(history[-20:]) if history else 0.0
            
            current_log_status.append(f"{group}_r: {rolling_mean:.3f}")

        pbar.set_description(" | ".join(current_log_status))
        pbar.update() 
    

if __name__ == "__main__":
    main()