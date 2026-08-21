import json
import argparse
import os
import copy
import tempfile
import numpy as np
import pandas as pd
from tensordict import nn
import torch
from pathlib import Path
from tqdm import tqdm

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.collectors import Collector
from torchrl.envs import ExplorationType, InitTracker, ObservationNorm, PettingZooEnv, Transform, TransformedEnv, PettingZooWrapper, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhDelta, OrnsteinUhlenbeckProcessModule
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

import sys 
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from network_env.network_env_v4 import NetworkEnvV4

class SafeMADDPGTrainer:
    
    def __init__(
        self,
        environment: NetworkEnvV4 = None,
        n_agent = 1,
        frames_per_batch = 100,
        n_iter = 100,
        min_replay_size = 1000,
        memory_size = 100000,
        n_optimizer_steps = 100,
        train_batch_size = 128,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        max_grad_norm = 1.0,
        polyak_tau = 0.005,
        gamma = 0.99,
        critic_configs = None,
        actor_configs = None,
        noise_configs = None,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path = None,
        seed = 0,
        random_frames = 0,
        noise_sigma = 0.2
    ):
        
        self.n_agent = n_agent
        self.frames_per_batch = frames_per_batch
        self.n_iter = n_iter
        self.min_replay_size = min_replay_size
        self.memory_size = memory_size
        self.n_optimizer_steps = n_optimizer_steps
        self.train_batch_size = train_batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_grad_norm = max_grad_norm
        self.polyak_tau = polyak_tau
        self.gamma = gamma
        self.device = device
        self.critic_configs = critic_configs if critic_configs is not None else {}
        self.actor_configs = actor_configs if actor_configs is not None else {}
        self.noise_configs = noise_configs if noise_configs is not None else {}
        self.save_path = save_path
        self.random_frames = random_frames
        self.noise_sigma = noise_sigma
        
        torch.manual_seed(seed)

        # Environment setup
        if environment is not None:
            self.env = PettingZooWrapper(env = environment, use_mask = True, categorical_actions=False, device = self.device)
            self.env = TransformedEnv(self.env, InitTracker())
            
            obs_key = ("slice", "observation")

            # Instantiate the observation normalization transform
            #obs_norm = ObservationNorm(
            #    in_keys=[obs_key],
            #    standard_normal=True,
            #)
            
            #self.env.append_transform(obs_norm)
            
            #self.env.transform[1].init_stats(100)
        else:
            raise ValueError("Environment must be provided for MADDPGTrainer.")
    
    def train(self):

        total_frames = self.n_iter * self.frames_per_batch
        
        # Networks
        policy_modules = {}
        policies = {}
        exploration_policies = {}
        critics = {}
        
        env = self.env
        
        # ACTOR NETWORKS
        
        for group, agents in env.group_map.items():
            policy_net = MultiAgentMLP(
                n_agent_inputs = env.observation_spec[group, "observation"].shape[-1],
                n_agent_outputs = env.full_action_spec[group, "action"].shape[-1],
                n_agents = len(agents),
                centralized = False,
                share_params = self.actor_configs.get("share_parameter", False),
                device = self.device,
                depth = self.actor_configs.get("depth", 2),
                num_cells = self.actor_configs.get("num_cells", 64),
                activation_class = self.actor_configs.get("activation_class", torch.nn.ReLU),
                #norm_class=lambda: torch.nn.LayerNorm(normalized_shape=self.critic_configs.get("num_cells", 64))
                
            )
            
            policy_module = TensorDictModule(
                module = policy_net,
                in_keys = [(group, "observation")], 
                out_keys = [(group, "param")]
                
            )
            
            policy_modules[group] = policy_module
            
            policy = TensorDictSequential(
                        ProbabilisticActor(
                            module=policy_modules[group],
                            spec=env.full_action_spec[group,"action"],
                            in_keys=[(group,"param")],
                            out_keys=[(group,"action_raw")],
                            distribution_class=TanhDelta,
                            distribution_kwargs={
                                "low": env.full_action_spec_unbatched[group, "action"].space.low,
                                "high": env.full_action_spec_unbatched[group, "action"].space.high
                            },
                            return_log_prob = False
                        ),

                        TensorDictModule(
                            ActionProjectionModule(),
                            in_keys=[(group,"action_raw")],
                            out_keys=[(group,"action")]
                        )
                    )
            policy.spec = env.full_action_spec[group,"action"]
            
            #policy = ProbabilisticActor(
            #    module = policy_modules[group],
            #    spec = env.full_action_spec[group, "action"],
            #    in_keys = [(group, "param")],
            #    out_keys = [(group, "action")],
            #    distribution_class = TanhDelta,
            #    distribution_kwargs = {
            #        "low": env.full_action_spec_unbatched[group, "action"].space.low,
            #        "high": env.full_action_spec_unbatched[group, "action"].space.high
            #    },
            #    return_log_prob = False
            #)
            
            policies[group] = policy 
            
            exploration_policy = TensorDictSequential(
                policy,
                OrnsteinUhlenbeckProcessModule(
                    spec = policy.spec.clone(),
                    annealing_num_steps = total_frames // 2,
                    action_key = (group, "action"),
                    sigma = self.noise_sigma,
                ).to(self.device)
            )
            
            exploration_policies[group] = exploration_policy
        
            # CRITIC NETWORKS
            cat_module = TensorDictModule(
                lambda obs, action: torch.cat([obs, action], dim = -1), 
                in_keys = [(group, "observation"), (group, "action")],
                out_keys = [(group, "obs_action")]
            )
            critic_net = MultiAgentMLP(
                n_agent_inputs = env.observation_spec[group, "observation"].shape[-1] + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs = 1,
                n_agents = self.n_agent,
                centralized = self.critic_configs.get("centralized_critic", True),
                share_params = self.critic_configs.get("share_parameter", False),
                device = self.device,
                depth = self.critic_configs.get("depth", 2),
                num_cells = self.critic_configs.get("num_cells", 64),
                activation_class = self.critic_configs.get("activation_class", torch.nn.ReLU),
                #norm_class=lambda: torch.nn.LayerNorm(normalized_shape=self.critic_configs.get("num_cells", 64))
            )
            critic_module = TensorDictModule(
                module = critic_net,
                in_keys = [(group, "obs_action")],
                out_keys =[(group, "state_action_value")]
            )
                
            critics[group] = TensorDictSequential(cat_module, critic_module)
            
        # COLLECTOR
        
        agents_exploration_policy = TensorDictSequential(*exploration_policies.values())
        collector = Collector(
            env,
            agents_exploration_policy,
            device = self.device,
            frames_per_batch = self.frames_per_batch,
            total_frames = total_frames,
            init_random_frames=self.random_frames
        )
        
        # REPLAY BUFFER, LOSSES, OPTIMIZERS
        replay_buffers = {}
        losses = {}
        
        for group in env.group_map.keys():
            scratch_dir = tempfile.TemporaryDirectory().name
            
            storage = LazyMemmapStorage(
                self.memory_size,
                scratch_dir = scratch_dir,
            )
            
            replay_buffer = ReplayBuffer(
                storage = storage,
                sampler = RandomSampler(),
                batch_size = self.train_batch_size
            )
            
            if self.device.type != "cpu":
                replay_buffer.append_transform(lambda x: x.to(self.device))
                
            replay_buffers[group] = replay_buffer
            
            loss_module = DDPGLoss(
                actor_network = policies[group],
                value_network = critics[group],
                delay_value = True,
                loss_function = "l2"
            )
            loss_module.set_keys(
                state_action_value = (group, "state_action_value"),
                reward = (group, "reward"),
                done = (group, "done"),
                terminated = (group, "terminated"),
            )
            loss_module.make_value_estimator(
                ValueEstimators.TD0,
                gamma = self.gamma
            )
            
            losses[group] = loss_module
        
        target_updaters = {group: SoftUpdate(loss, tau = self.polyak_tau) for group, loss in losses.items()}
        optimizers = {
            group:{
                "loss_actor": torch.optim.Adam(loss.actor_network.parameters(), lr = self.actor_lr),
                "loss_value": torch.optim.Adam(loss.value_network.parameters(), lr = self.critic_lr)
            } for group, loss in losses.items()
        }
        
        # TRAINING LOOP
        pbar = tqdm(total = self.n_iter)
        reward_history_map = {group: [] for group in env.group_map.keys()}
        iteration_rewards = {group: [] for group in env.group_map.keys()}
        #incomplete_transitions = {group: {} for group in env.group_map.keys()}
        train_group_map = copy.deepcopy(env.group_map)

        #synchronized_timer = 0
        
        for iteration, batch in enumerate(collector):
            current_frames = batch.numel()
            batch = process_batch(batch, train_group_map)
            for group in train_group_map.keys():
                group_batch = batch.exclude(
                    *[
                        key
                        for _group in env.group_map.keys()
                        if _group != group
                        for key in [_group, ("next", _group)]
                    ]
                )  # Exclude data from other groups
                group_batch = group_batch.reshape(
                    -1
                )  # This just affects the leading dimensions in batch_size of the tensordict
                replay_buffers[group].extend(group_batch)
                
                rewards = batch.get(("next",group ,"reward"))
                mean_reward = rewards.mean().item()
                iteration_rewards[group].append(mean_reward)
            
                if len(replay_buffers[group]) >= self.min_replay_size:
                    #torch.autograd.set_detect_anomaly(True)
                    #print("OPTIMIZATION")
                    for _ in range(self.n_optimizer_steps):
                        subdata = replay_buffers[group].sample()
                        loss_vals = losses[group](subdata)
                        
                        for loss_name in ["loss_actor","loss_value"]:
                            loss = loss_vals[loss_name]
                            optimizer = optimizers[group][loss_name]
                            
                            #if loss_name == 'loss_value':
                            #    print(f'Critic loss : {loss.item()}')
                            
                            loss.backward()
                            
                            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], self.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                    
                        target_updaters[group].step()     
                    exploration_policies[group][-1].step(current_frames)
            
            # Progress update
            current_log_status = []
            for group in env.group_map.keys():
                if iteration_rewards[group]:
                    avg_reward_this_step = sum(iteration_rewards[group]) / len(iteration_rewards[group])
                    reward_history_map[group].append(avg_reward_this_step)
                    iteration_rewards[group] = []
                history = reward_history_map[group]
                rolling_mean = sum(history[-20:]) / len(history[-20:]) if history else 0.0
                current_log_status.append(f"{group}_r: {rolling_mean:.3f}")

        pbar.set_description(" | ".join(current_log_status))
        pbar.update()
        
      
        for group, policy in policies.items():
            model_filename = os.path.join(self.save_path, f"maddpg_actor_{group}.pt")
            try:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path, exist_ok=True)
            except Exception as e:
                print("Fail to save result")

            torch.save(policy.state_dict(), model_filename)
            print(f"Saved trained policy weights for group '{group}' to {model_filename}")
            

        return 
            
                           
                    
class SafeMADDPGTester:
    
    def __init__(
        self,
        environment: NetworkEnvV4 = None,
        n_agent = 1,
        frames_per_batch = 100,
        n_iter = 100,
        min_replay_size = 1000,
        memory_size = 100000,
        n_optimizer_steps = 100,
        train_batch_size = 128,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        max_grad_norm = 1.0,
        polyak_tau = 0.005,
        gamma = 0.99,
        critic_configs = None,
        actor_configs = None,
        noise_configs = None,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_path = None,
    ):
    
        self.n_agent = n_agent
        self.frames_per_batch = frames_per_batch
        self.n_iter = n_iter
        self.min_replay_size = min_replay_size
        self.memory_size = memory_size
        self.n_optimizer_steps = n_optimizer_steps
        self.train_batch_size = train_batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.max_grad_norm = max_grad_norm
        self.polyak_tau = polyak_tau
        self.gamma = gamma
        self.device = device
        self.critic_configs = critic_configs if critic_configs is not None else {}
        self.actor_configs = actor_configs if actor_configs is not None else {}
        self.noise_configs = noise_configs if noise_configs is not None else {}
        self.load_path = load_path
        
        
        # Environment setup
        if environment is not None:
            self.env = PettingZooWrapper(env = environment, use_mask = True, categorical_actions=False, device = self.device)
            self.env = TransformedEnv(self.env, InitTracker())

        else:
            raise ValueError("Environment must be provided for MADDPGTrainer.")
    
    def test(self):
        total_frames = self.n_iter * self.frames_per_batch
        
        # Networks
        policy_modules = {}
        policies = {}
        
        env = self.env
        
        # ACTOR NETWORKS
        
        for group, agents in env.group_map.items():
            policy_net = MultiAgentMLP(
                n_agent_inputs = env.observation_spec[group, "observation"].shape[-1],
                n_agent_outputs = env.full_action_spec[group, "action"].shape[-1],
                n_agents = len(agents),
                centralized = False,
                share_params = self.actor_configs.get("share_parameter", False),
                device = self.device,
                depth = self.actor_configs.get("depth", 2),
                num_cells = self.actor_configs.get("num_cells", 64),
                activation_class = self.actor_configs.get("activation_class", torch.nn.ReLU),
                
            )
            
            policy_module = TensorDictModule(
                module = policy_net,
                in_keys = [(group, "observation")], 
                out_keys = [(group, "param")]
                
            )
            
            policy_modules[group] = policy_module
            
            policy = TensorDictSequential(
                        ProbabilisticActor(
                            module=policy_modules[group],
                            spec=env.full_action_spec[group,"action"],
                            in_keys=[(group,"param")],
                            out_keys=[(group,"action_raw")],
                            distribution_class=TanhDelta,
                            distribution_kwargs={
                                "low": env.full_action_spec_unbatched[group, "action"].space.low,
                                "high": env.full_action_spec_unbatched[group, "action"].space.high
                            },
                            return_log_prob = False
                        ),

                        TensorDictModule(
                            ActionProjectionModule(),
                            in_keys=[(group,"action_raw")],
                            out_keys=[(group,"action")]
                        )
                    )
            policy.spec = env.full_action_spec[group,"action"]

            
            #policy = ProbabilisticActor(
            #    module = policy_modules[group],
            #    spec = env.full_action_spec[group, "action"],
            #    in_keys = [(group, "param")],
            #    out_keys = [(group, "action")],
            #    distribution_class = TanhDelta,
            #   distribution_kwargs = {
            #        "low": env.full_action_spec_unbatched[group, "action"].space.low,
            #        "high": env.full_action_spec_unbatched[group, "action"].space.high
            #    },
            #    return_log_prob = False
            #)
            
            
            # Load the Saved Checkpoint
            model_weight_path = os.path.join(self.load_path, f"maddpg_actor_{group}.pt")
            if os.path.exists(model_weight_path):
                policy.load_state_dict(torch.load(model_weight_path, map_location=self.device))
                print(f"Successfully loaded weights for group '{group}' from {model_weight_path}")
            else:
                print(f"Warning: Checkpoint not found at {model_weight_path}. Executing with random weights!")
                
            policies[group] = policy
            
        # Execution Policy (No OU Process Module attached -> Deterministic actions)
        eval_policy = TensorDictSequential(*policies.values())
        collector = Collector(env, eval_policy, device=self.device, frames_per_batch=self.frames_per_batch, total_frames=total_frames)

        print("Starting Deterministic Evaluation...")
        synchronized_timer = 0
        incomplete_transitions = {group: {} for group in env.group_map.keys()}

        # Force Deterministic Execution Mode (ExplorationType.MODE selects the mode/mean of distribution)
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            for iteration, batch in enumerate(tqdm(collector)):
                
                # Map out local incomplete slots for evaluation tracking
                for idx, time in enumerate(range(synchronized_timer, synchronized_timer + self.frames_per_batch)):
                    single_step_td = batch[idx]
                    for group in env.group_map.keys():
                        group_data = single_step_td.exclude(*[key for _group in env.group_map.keys() if _group != group for key in [_group, ("next", _group)]])
                        incomplete_transitions[group][time] = group_data.clone().reshape(-1)

                # Pull evaluations synchronously
                for group in env.group_map.keys():
                    if env.is_ready():
                        ready_reward = env.get_ready_reward()
                        for time, reward in sorted(ready_reward.items()):
                            if time in incomplete_transitions[group]:
                                incomplete_transitions[group].pop(time)

                synchronized_timer += self.frames_per_batch


        return
        
        
    
    

def process_batch(batch: TensorDictBase, group_map) -> TensorDictBase:
    for group in group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        
        if nested_done_key not in keys:
            batch.set(nested_done_key, batch.get(("next", "done"))).unsqueeze(-1).expand(*group_shape, 1)
        if nested_terminated_key not in keys:
            batch.set(nested_terminated_key, batch.get(("next", "terminated"))).unsqueeze(-1).expand(*group_shape, 1)

    return batch



class BoundedOrthogonalProjection(torch.nn.Module):
    def __init__(self, in_keys_inv=None, out_keys_inv=None, threshold=-0.575 / 0.475, inequality=True, num_iters=15):
        if in_keys_inv is None:
            in_keys_inv = [("slice", "action")]
        if out_keys_inv is None:
            out_keys_inv = in_keys_inv
        
        super().__init__()
            
        self.threshold = threshold
        self.inequality = inequality
        self.num_iters = num_iters

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [n_agents, action_dim] (or any shape with arbitrary leading batch dims)
        
        # 1. Establish search brackets of shape [1, action_dim]
        # max across n_agents (dim=-2) keeps dimensions -> shape: [1, action_dim]
        high = action.max(dim=-2, keepdim=True).values + 1.0
        
        if self.inequality:
            # Match high's shape exactly: [1, action_dim]
            low = torch.zeros_like(high)
            
            # Check if initial clamped actions already satisfy sum <= threshold
            initial_sum = torch.clamp(action, -1.0, 1.0).sum(dim=-2, keepdim=True)
            already_valid = initial_sum <= self.threshold
            
            # If valid, set high = 0 so final lambda remains exactly 0
            high = torch.where(already_valid, low, high)
        else:
            low = action.min(dim=-2, keepdim=True).values - 1.0

        # 2. Vectorized Bisection Loop
        for _ in range(self.num_iters):
            mid = (low + high) / 2.0  # Shape: [1, action_dim]
            
            # [n_agents, action_dim] - [1, action_dim] broadcasts cleanly
            projected = torch.clamp(action - mid, min=-1.0, max=1.0)
            
            # Sum across n_agents -> Shape: [1, action_dim]
            current_sum = projected.sum(dim=-2, keepdim=True)
            
            mask = current_sum > self.threshold
            low = torch.where(mask, mid, low)
            high = torch.where(mask, high, mid)
            
        # 3. Final projection
        final_lambda = (low + high) / 2.0
        #print(f"Final lambda: {final_lambda}")
        return torch.clamp(action - final_lambda, min=-1.0, max=1.0)


class ActionProjectionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.projector = BoundedOrthogonalProjection(
            threshold=-0.575 / 0.475,
            inequality=True,
            num_iters=15
        )

    def forward(self, action):
        return self.projector.forward(action)
    
