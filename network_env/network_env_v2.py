from collections import deque
import copy
import functools
import random
import numpy as np
import json
import os
import logging
from pathlib import Path
from gymnasium.spaces import Box 
from pettingzoo import ParallelEnv
import pandas as pd

# Module-level logger (will be configured per instance)
logger = logging.getLogger(__name__)

TEST_MODE = False  # Set to True to enable test-specific logging behavior

class NetworkEnvV2(ParallelEnv):
    
    def __init__(self, n_slices=1, resource_path=None, traffic_path=None, log_path=None,
                 resource_scaling_factor=1.0, scheduler=None, test_demand=None, alpha=1.0, beta=100.0):
        self.n_slices = n_slices
        self.resource_scaling_factor = resource_scaling_factor
        self.scheduler = scheduler if scheduler is not None else FIFOScheduler()
        self.log_path = log_path
        self.traffic_path = traffic_path
        self.test_demand = test_demand
        self.alpha = alpha
        self.beta = beta
        # Configure logging with log_path
        self._setup_logging()
        
        logger.debug(f"[__init__] n_slices={n_slices}, resource_scaling_factor={resource_scaling_factor}, scheduler={self.scheduler.__class__.__name__}")
        
        self._read_config(resource_path)
        
        self.agents = [f'slice_{i}' for i in range(n_slices)]
        self.possible_agents = self.agents[:]
        
        self.slices = {}
        for agent in self.agents:
            self.slices[agent] = Slice(slice_id=agent, resource=self.resources)
        
        self.current_time = 0
        #self.alpha = 1.0
        #self.beta = 100.0
        # Structure: { arrival_time_t_prime: { 'slice_0': reward_val, 'slice_1': reward_val } }
        self._ready_rewards_ledger = {}
        self.recorders = {agent: Recorder(self.slices[agent]) for agent in self.agents}
        
        #logger.debug(f"[__init__] initialized: agents={self.agents}, current_time={self.current_time}, alpha={self.alpha}, beta={self.beta}")
        #logger.debug(f"[__init__] resources={list(self.resources.keys())}")
    
    def set_test_mode(self, test_mode=True):
        global TEST_MODE
        TEST_MODE = test_mode
        logger.disabled = not TEST_MODE
        if TEST_MODE:
            self._setup_logging()
            logger.setLevel(logging.DEBUG)

    def _setup_logging(self):
        """Configure logging to write to log_path directory if specified."""
        # Clear existing handlers
        logger.handlers.clear()
        logger.disabled = not TEST_MODE
        if not TEST_MODE:
            return
        
        # Determine log file path
        if self.log_path:
            log_dir = Path(self.log_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / 'network_env_debug.log'
        else:
            log_file = Path('network_env_debug.log')
        
        # Configure logging only to file when in test mode
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)
            ],
            force=True
        )
        
        logger.info(f"[_setup_logging] Logging initialized, log file: {log_file}")
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new training episode.
        Matches the PettingZoo ParallelEnv specification.
        """
        # 1. Handle seeding for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # If using torch in your env setup:
            # torch.manual_seed(seed)
            
        # 2. Reset the clock to the initial discrete time slot
        self.current_time = 0
        logger.debug(f"[reset] current_time set to {self.current_time}")
        
        # 3. Restore all physical resource capacities back to maximum capacity
        for res in self.resources.values():
            res.reset()
        logger.debug(f"[reset] reset resources: {[(r.resource_id, r.available_capacity) for r in self.resources.values()]}")
            
        # 4. Clear out active task queues and wipe out the asynchronous ledger history
        for agent in self.agents:
            self.slices[agent].task_queue.clear()
        logger.debug(f"[reset] cleared task queues for all agents")
            
        self._ready_rewards_ledger.clear()
        logger.debug(f"[reset] cleared rewards ledger")
        
        self.recorders = {agent: Recorder(self.slices[agent]) for agent in self.agents}
        
        self.demand = {}
        for agent in self.agents:
            if self.test_demand is not None:
                self.demand[agent] = self.test_demand
            else:
                self.demand[agent] = pd.read_csv(os.path.join(self.traffic_path, f'{agent}_demand.csv'))

        
        # 5. Build and return initial observations and info mappings
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def _get_obs(self, agent):
        """
        Constructs the state observation vector for a specific network slice/agent.
        
        The observation includes:
          - Total outstanding CPU backlog across all active task batches for each server.
          - Total outstanding bandwidth backlog across all active task batches for each link.
          - The slice preference profiles (lambda_i, rho_i) to guide multi-objective learning.
        """
        slice_obj = self.slices[agent]
        
        # Initialize workload accumulation maps for all tracking resources in this slice
        accumulated_demands = {res_id: 0.0 for res_id in slice_obj.idx_to_resource}
        
        if self.test_demand is not None:
            new_task = Task(arrival_time=self.current_time, resource_demand={
                res_id: self.test_demand * self.resources[res_id].capacity
                for res_id in slice_obj.idx_to_resource
            })
            
        else:
            new_task = Task(arrival_time=self.current_time, resource_demand={
                res_id: self.demand[agent].iloc[self.current_time][res_id]
                for res_id in slice_obj.idx_to_resource
            })
            
        slice_obj.add_task(new_task)
        # Aggregate remaining work over all task batches currently waiting in the pipeline
        for task in slice_obj.task_queue:
            for res_id, remaining_demand in task.resource_demand.items():
                if res_id in accumulated_demands:
                    accumulated_demands[res_id] += float(remaining_demand)

                        
        logger.debug(f"[_get_obs] agent={agent}, accumulated_demands={accumulated_demands}")
        
        for res_id in slice_obj.idx_to_resource:
            self.recorders[agent].add_accumulated(res_id, accumulated_demands[res_id])
        
        for res_id in slice_obj.idx_to_resource:
            self.recorders[agent].add_available_capacity(res_id, self.resources[res_id].available_capacity)
                    
        # Construct the features vector following a predictable indexing sequence
        obs_features = []
        for res_id in slice_obj.idx_to_resource:
            obs_features.append(accumulated_demands[res_id] / self.resources[res_id].capacity)  # Normalize backlog by capacity for better learning stability
        
        for res_id in slice_obj.idx_to_resource:
            obs_features.append(float(self.resources[res_id].available_capacity) / self.resources[res_id].capacity)  # Normalize available capacity by maximum capacity
        
        # Append the slice priority preferences (Equation 15 & 16 weights)
        obs_features.append(float(slice_obj.lambda_pref))  # Latency priority
        obs_features.append(float(slice_obj.rho_pref))     # Energy priority
        
        logger.debug(f"[_get_obs] agent={agent}, obs_features={obs_features}")
        # Convert to a flat float32 array suitable for standard neural network ingestion
        return np.array(obs_features, dtype=np.float32)

        
    def step(self, actions):
        logger.debug(f"[step] current_time={self.current_time}, actions={actions}")
        
        items = list(actions.items())
        random.shuffle(items)
        
        if TEST_MODE:
            rewards = {agent: 0.0 for agent in self.agents}

                
        for res in self.resources.values():
            res.reset()
            
        # 1. Gather all requested allocations to enforce GLOBAL constraints (Eq 11)
        requested_allocations = {res_id: 0.0 for res_id in self.resources}
        agent_requests = {agent: {} for agent in self.agents}
        
        for agent, action in actions.items():
            slice_obj = self.slices[agent]
            recorder = self.recorders[agent]
            
            for idx, res_id in enumerate(slice_obj.idx_to_resource):
                resource = self.resources[res_id]
                #normalized_action = (action[idx] + 1.0) / 2.0 
                #normalized_action = 0.3 * action[idx] + 0.7
                normalized_action = action[idx]*0.475 + 0.525
                req = normalized_action * slice_obj.resource[res_id].capacity * self.resource_scaling_factor
                recorder.add_action(res_id, normalized_action)
                
                if req > resource.available_capacity:
                    adjusted_req = resource.available_capacity - 0.05 * resource.capacity
                    req = max(0.0, adjusted_req)
                
                resource.allocate(req)

                agent_requests[agent][res_id] = req
                requested_allocations[res_id] += req
        
        #logger.debug(f"[step] agent_requests={agent_requests}, requested_allocations={requested_allocations}")
                
        # 2. Resolve global capacities and compute actual total allocations
        actual_allocations = {agent: {} for agent in self.agents}
        total_actual_allocations = {res_id: 0.0 for res_id in self.resources}
        
        #print(actual_allocations)
        
        for res_id, resource in self.resources.items():
            total_req = requested_allocations[res_id]
            
            # If total demand exceeds physical capacity, scale down proportionally
            #scale = resource.capacity / total_req if total_req > resource.capacity else 1.0
            scale = 1.0 
                
            for agent in self.agents:
                if res_id in agent_requests[agent]:
                    alloc = agent_requests[agent][res_id] * scale
                    actual_allocations[agent][res_id] = alloc
                    total_actual_allocations[res_id] += alloc

        for agent, allocations in actual_allocations.items():
            recorder = self.recorders[agent]
            for res_id, allocation in allocations.items():
                recorder.add_allocation(res_id, float(allocation))
        
        logger.debug(f"[step] actual_allocations={actual_allocations}, total_actual_allocations={total_actual_allocations}")
                    
            
        # 3. Compute Power f(U_m(k)) for all servers
        resource_powers = {}
        for res_id, resource in self.resources.items():
            u_m = total_actual_allocations[res_id] / resource.capacity if resource.capacity > 0 else 0
            #if u_m > 1.0 or u_m < 1e-5:
            #    raise ValueError(f"Utilization out of bounds for resource {res_id}: u_m={u_m}, total_actual_allocations={total_actual_allocations[res_id]}, capacity={resource.capacity} at time={self.current_time}")
            resource_powers[res_id] = resource.power_function(u_m)
        
        logger.debug(f"[step] resource_powers={resource_powers}")

        # 4. Schedule and apply proportional energy
        for agent in self.agents:
            slice_obj = self.slices[agent]
            
            for res_id, allocated_amount in actual_allocations[agent].items():
                self.scheduler.schedule(
                    task_queue=slice_obj.task_queue, 
                    res_id=res_id, 
                    allocated_amount=allocated_amount, 
                    total_resource_allocation=total_actual_allocations[res_id],
                    server_power=resource_powers[res_id],
                    current_time=self.current_time
                )
                            
            # 5. Check for Completions & Rewards
            completed_tasks = []
            for task in slice_obj.task_queue:
                if task.is_complete():
                    completed_tasks.append(task)
                    
                    end_to_end_latency = self.current_time - task.arrival_time + 1
                    lambda_i, rho_i = slice_obj.lambda_pref, slice_obj.rho_pref
                    
                    reward_task = (lambda_i * (self.alpha / end_to_end_latency)) + \
                                  (rho_i * (self.beta / max(1e-5, task.accumulated_energy)))
                    
                    #reward_task = - end_to_end_latency
                    
                    recorder.add_latency(task.arrival_time, end_to_end_latency)
                    recorder.add_energy(task.arrival_time, float(task.accumulated_energy))
                    #recorder.add_reward(reward_task)
                    
                    #if end_to_end_latency > 5.0:
                    #    reward_task -= 1

                    
                    if TEST_MODE:
                        rewards[agent] += reward_task
                    
                    logger.debug(f"[step] agent={agent}, task_arrival={task.arrival_time}, end_to_end_latency={end_to_end_latency}, accumulated_energy={task.accumulated_energy}, reward_task={reward_task}")
                    #print(f"[step] time={self.current_time}, agent={agent}, task_arrival={task.arrival_time}, latency={end_to_end_latency}, energy={task.accumulated_energy:.4f}, reward={reward_task:.4f}")
                    t_prime = task.arrival_time
                    
                    if t_prime not in self._ready_rewards_ledger:
                        self._ready_rewards_ledger[t_prime] = {}
                    self._ready_rewards_ledger[t_prime][agent] = reward_task
                    #print(f"[step] Recorded reward for agent={agent}, task_arrival={t_prime}, reward={reward_task}")
            
            for task in completed_tasks:
                slice_obj.task_queue.remove(task)
                
        self.current_time += 1
        logger.debug(f"[step] incremented current_time to {self.current_time}")
        
        if not TEST_MODE:
            rewards = {agent: 0.0 for agent in self.agents}
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        #logger.debug(f"[step] rewards={rewards}, terminations={terminations}")

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        logger.debug(f"[step] observations computed, returning step results")
        return observations, rewards, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        number_of_resources = self.slices[agent].number_of_resources()
        # [Available Capacities] + [Queued Demands] + [Lambda, Rho]
        return Box(low=0.0, high=np.inf, shape=(number_of_resources * 2 + 2,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        number_of_resources = self.slices[agent].number_of_resources()
        return Box(low=-1.0, high=1.0, shape=(number_of_resources,), dtype=np.float32)

    def _read_config(self, resource_path):
        if not resource_path or not os.path.exists(resource_path):
            # Fallback mock configuration for testing if no file is provided
            config = [
                {"type": "mec", "id": 1, "resources": {"cpu": 100}},
                {"type": "link", "id": 1, "resources": {"bandwidth": 50}}
            ]
            logger.debug(f"[_read_config] using default mock config, resource_path={resource_path}")
        else:
            with open(resource_path, 'r') as f:
                config = json.load(f)
            logger.debug(f"[_read_config] loaded config from {resource_path}")
        
        self.resources = {}
            
        for i, item in enumerate(config):
            if item['type'] == 'mec':
                resource_id = item['type'] + '_' + str(item['id'])
                self.resources[resource_id] = Resource(resource_id, item['resources']['cpu'], resource_type='mec')
            elif item['type'] == 'link':
                resource_id = item['type'] + '_' + str(item['id'])
                self.resources[resource_id] = Resource(resource_id, item['resources']['bandwidth'], resource_type='link')
        
        logger.debug(f"[_read_config] initialized resources: {list(self.resources.keys())}")
    
    
    def is_ready(self):
        """
        Signals to the training loop that at least one arrival time has
        completed reward values for every slice.
        """
        for time_step, rewards in self._ready_rewards_ledger.items():
            if len(rewards) == len(self.agents):
                return True
        return False

    def get_ready_reward(self):
        """
        Flushes and returns only fully-complete reward entries where every
        slice has a reward for the same arrival timestamp.
        
        Returns:
            dict: { time_step: mean_reward }
        """
        ready = {}
        completed_time_steps = []
        for time_step, rewards in self._ready_rewards_ledger.items():
            if len(rewards) == len(self.agents):
                mean_reward = sum(rewards.values()) / len(rewards)
                ready[time_step] = {agent: mean_reward for agent, reward in rewards.items()}
                completed_time_steps.append(time_step)
                logger.debug(f"[get_ready_reward] time_step={time_step}, rewards={rewards}, mean_reward={mean_reward}")
                for agent, _ in rewards.items():
                    self.recorders[agent].add_reward(time_step, mean_reward)
                    #print(f"[get_ready_reward] Recorded mean reward for agent={agent}, time_step={time_step}, mean_reward={mean_reward}")

        for time_step in completed_time_steps:
            del self._ready_rewards_ledger[time_step]

        return ready

    def save_statistics(self, output_dir=None):
        """Save collected statistics using per-slice Recorder objects."""
        if output_dir is None:
            output_dir = self.log_path or '.'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for agent, recorder in self.recorders.items():
            recorder.save_result(str(output_path / agent))

        logger.info(f"[save_statistics] recorder statistics saved to {output_path}")


class Recorder():
    def __init__(self, slice_obj):
        self.slice_id = slice_obj.slice_id
        self.idx_to_resource = slice_obj.idx_to_resource

        self.action = {resource: [] for resource in self.idx_to_resource}
        self.allocation = {resource: [] for resource in self.idx_to_resource}
        self.latency = {}
        self.energy = {}
        self.reward = {}
        self.rejection = {resource: [] for resource in self.idx_to_resource}
        self.accumulated = {resource: [] for resource in self.idx_to_resource}
        self.available_capacity = {resource: [] for resource in self.idx_to_resource}
    
    def add_available_capacity(self, id, available_capacity):
        self.available_capacity[id].append(available_capacity)

    def add_accumulated(self, id, accumulated):
        self.accumulated[id].append(accumulated)

    def add_action(self, id, action):
        self.action[id].append(action)

    def add_allocation(self, id, allocation):
        self.allocation[id].append(allocation)

    def add_latency(self,t_prime, latency):
        self.latency[t_prime] = latency
    
    def add_energy(self, t_prime, energy):
        self.energy[t_prime] = energy
    
    def add_rejection(self, id, rejection):
        self.rejection[id].append(rejection)
    
    def add_reward(self, t_prime,reward):
        self.reward[t_prime] = reward
    
    def save_result(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            
            # Convert dictionaries to lists by extracting values (self.reward/latency/energy are dicts with t_prime as keys)
            reward = {'reward': list(self.reward.values())}
            latency = {'latency': list(self.latency.values())}
            energy = {'energy': list(self.energy.values())}
            #print(f"[Recorder.save_result] reward={reward}")
            data_to_save = {
                'action.csv': self.action,
                'allocation.csv': self.allocation,
                'latency.csv': latency,
                'energy.csv': energy,
                'rejection.csv': self.rejection,
                'reward.csv': reward,
                'accumulated.csv': self.accumulated
            }

            for filename, data in data_to_save.items():
                if data is not None and len(data) > 0:
                    df = pd.DataFrame(data)
                    #print(df)
                    full_path = os.path.join(path, filename)
                    df.to_csv(full_path, index=False)

            logger.debug(f"[Recorder.save_result] saved recorder data to {path}")
            print(f"[Recorder.save_result] Saved {len(self.reward)} rewards, {len(self.latency)} latencies, {len(self.energy)} energies to {path}")
        except Exception as e:
            logger.error(f"[Recorder.save_result] failed saving recorder data to {path}: {e}")
            print(f"[Recorder.save_result] ERROR: Failed to save recorder data to {path}: {e}")


class Resource():
    def __init__(self, resource_id, capacity, resource_type):
        self.resource_id = resource_id
        self.capacity = float(capacity)
        self.available_capacity = float(capacity)
        self.resource_type = resource_type # 'mec' or 'link'
        logger.debug(f"[Resource.__init__] resource_id={resource_id}, capacity={self.capacity}, type={resource_type}")

    def power_function(self, utilization):

        # According to Eq 9, energy is strictly summed over m in M_i (MEC servers)
        if self.resource_type == 'link':
            return 0.0 
            
        return 43.4779 * np.log(100 * utilization) + 226.8324 if np.log(100 * utilization) > 0 else 226.8324

    def allocate(self, amount):
        if amount <= self.available_capacity:
            self.available_capacity -= amount
            logger.debug(f"[Resource.allocate] resource_id={self.resource_id}, allocated={amount}, remaining_capacity={self.available_capacity}")
            return True
        #self.available_capacity = 0
        #logger.debug(f"[Resource.allocate] resource_id={self.resource_id}, allocation failed, remaining_capacity={self.available_capacity}")
        return False

    #def release(self, amount):
    #    self.available_capacity = min(self.capacity, self.available_capacity + amount)
    #    logger.debug(f"[Resource.release] resource_id={self.resource_id}, released={amount}, available_capacity={self.available_capacity}")
    
    def reset(self):
        logger.debug(f"[Resource.reset] resource_id={self.resource_id}, capacity={self.capacity}")
        self.available_capacity = self.capacity
        
class Slice():
    def __init__(self, slice_id, resource):
        self.slice_id = slice_id
        self.resource = resource
        self.idx_to_resource = [res_id for res_id in self.resource.keys()]
        self.task_queue = deque()
        
        # Preference weights for latency vs energy
        self.lambda_pref = 0.5 
        self.rho_pref = 0.5
        
        logger.debug(f"[Slice.__init__] slice_id={slice_id}, idx_to_resource={self.idx_to_resource}, lambda_pref={self.lambda_pref}, rho_pref={self.rho_pref}")

    def number_of_resources(self):
        return len(self.resource)

    def add_task(self, task):
        self.task_queue.append(task)
        logger.debug(f"[Slice.add_task] slice_id={self.slice_id}, task_arrival_time={task.arrival_time}, queue_size={len(self.task_queue)}")

class Task():
    def __init__(self, arrival_time, resource_demand):
        self.arrival_time = arrival_time
        self.original_demand = copy.deepcopy(resource_demand)
        self.resource_demand = resource_demand # dict of res_id -> remaining demand
        self.completion_times = {} # dict tracking when specific demands hit 0
        self.accumulated_energy = 0.0
        
        logger.debug(f"[Task.__init__] arrival_time={arrival_time}, resource_demand={resource_demand}, accumulated_energy={self.accumulated_energy}")

    def is_complete(self):
        # A task is completely served when ALL its required demands reach zero (Eq 7)
        return all(demand <= 0 for demand in self.resource_demand.values())

from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self, task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time):
        """
        allocated_amount: g_im(k) for this specific slice.
        total_resource_allocation: sum of g_im(k) across ALL slices (Eq 10 denominator).
        server_power: f(U_m(k)) total power consumption of the server (Eq 9).
        """
        pass

class FIFOScheduler(BaseScheduler):
    def schedule(self, task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time):
        remaining_allocation = allocated_amount
        logger.debug(f"[FIFOScheduler.schedule] res_id={res_id}, allocated_amount={allocated_amount}, server_power={server_power}, current_time={current_time}")
        
        for task in task_queue:
            if remaining_allocation <= 0:
                break
            
            if task.resource_demand.get(res_id, 0) > 0:
                deduction = min(task.resource_demand[res_id], remaining_allocation) # This is g_{imt'}(k)
                task.resource_demand[res_id] -= deduction
                remaining_allocation -= deduction
                
                logger.debug(f"[FIFOScheduler.schedule] deduction={deduction}, remaining_allocation={remaining_allocation}, task_arrival_time={task.arrival_time}")
                
                # --- EXACT ENERGY CALCULATION (Eq 9 & 10) ---
                if total_resource_allocation > 0:
                    # omega_{imt'}(k) = g_{imt'}(k) / sum(g_im(k))
                    omega = deduction / total_resource_allocation 
                    # Energy = omega * f(U_m(k))
                    task.accumulated_energy += omega * server_power
                    logger.debug(f"[FIFOScheduler.schedule] omega={omega}, accumulated_energy={task.accumulated_energy}")
                
                # Mark completion time (Eq 5 & 6)
                if task.resource_demand[res_id] == 0 and res_id not in task.completion_times:
                    task.completion_times[res_id] = current_time
                    logger.debug(f"[FIFOScheduler.schedule] task completed for res_id={res_id} at time={current_time}")

class ProcessorSharingScheduler(BaseScheduler):
    def schedule(self, task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time):
        active_tasks = [t for t in task_queue if t.resource_demand.get(res_id, 0) > 0]
        
        logger.debug(f"[ProcessorSharingScheduler.schedule] res_id={res_id}, allocated_amount={allocated_amount}, active_tasks={len(active_tasks)}, current_time={current_time}")
        
        if not active_tasks or allocated_amount <= 0:
            logger.debug(f"[ProcessorSharingScheduler.schedule] no active tasks or no allocation, returning")
            return
            
        share = allocated_amount / len(active_tasks) # This is g_{imt'}(k) for each task
        logger.debug(f"[ProcessorSharingScheduler.schedule] share per task={share}")
        
        for task in active_tasks:
            deduction = min(task.resource_demand[res_id], share)
            task.resource_demand[res_id] -= deduction
            
            logger.debug(f"[ProcessorSharingScheduler.schedule] task_arrival_time={task.arrival_time}, deduction={deduction}, remaining_demand={task.resource_demand[res_id]}")
            
            # --- EXACT ENERGY CALCULATION (Eq 9 & 10) ---
            if total_resource_allocation > 0:
                omega = deduction / total_resource_allocation
                task.accumulated_energy += omega * server_power
                logger.debug(f"[ProcessorSharingScheduler.schedule] omega={omega}, accumulated_energy={task.accumulated_energy}")
            
            if task.resource_demand[res_id] == 0 and res_id not in task.completion_times:
                task.completion_times[res_id] = current_time
                logger.debug(f"[ProcessorSharingScheduler.schedule] task completed for res_id={res_id} at time={current_time}")