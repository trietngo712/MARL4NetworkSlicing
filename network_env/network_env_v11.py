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
PENALTY_REWARD = -1  # Large negative reward for queue overflow
MAX_QUEUE_LENGTH = 5  # Maximum allowed queue length before penalizing

END_EPISODE = False  # Flag to signal episode termination on queue overflow

class NetworkEnvV11(ParallelEnv):
    
    def __init__(self,
        n_slices=1, 
        resource_path=None, 
        traffic_path=None, 
        log_path=None,
        resource_scaling_factor=1.0, 
        scheduler=None, 
        test_demand=None,
        max_queue_length=MAX_QUEUE_LENGTH,
        penalty_reward=PENALTY_REWARD,
        latency_preference=None,
        energy_preference=None,
        alpha = 1.0,
        beta = 1000.0):
        
        self.n_slices = n_slices
        self.resource_scaling_factor = resource_scaling_factor
        self.scheduler = scheduler if scheduler is not None else FairShareScheduler()
        self.log_path = log_path
        self.traffic_path = traffic_path
        self.test_demand = test_demand
        self.max_queue_length = max_queue_length
        self.penalty_reward = penalty_reward
        
        # Configure logging with log_path
        self._setup_logging()
        
        logger.debug(f"[__init__] n_slices={n_slices}, resource_scaling_factor={resource_scaling_factor}, scheduler={self.scheduler.__class__.__name__}")
        
        self._read_config(resource_path)
        
        self.agents = [f'slice_{i}' for i in range(n_slices)]
        self.possible_agents = self.agents[:]
        
        self.slices = {}
        for i, agent in enumerate(self.agents):
            self.slices[agent] = Slice(slice_id=agent, resource=self.resources)
            self.slices[agent].set_preferences(latency_preference[i] if latency_preference is not None else 0.5, energy_preference[i] if energy_preference is not None else 0.5)
            
        
        self.current_time = 0
        self.alpha = alpha
        self.beta = beta
        # Structure: { arrival_time_t_prime: { 'slice_0': reward_val, 'slice_1': reward_val } }
        self._ready_rewards_ledger = {}
        self.recorders = {agent: Recorder(self.slices[agent]) for agent in self.agents}
        
        self.current_queue = {agent: {} for agent in self.agents}
        
        #logger.debug(f"[__init__] initialized: agents={self.agents}, current_time={self.current_time}, alpha={self.alpha}, beta={self.beta}")
        #logger.debug(f"[__init__] resources={list(self.resources.keys())}")
        
        self.previous_action = []
        self.current_action = []
        
        self.demand = {}
        for agent in self.agents:
            if self.test_demand is not None:
                self.demand[agent] = self.test_demand
            else:
                self.demand[agent] = pd.read_csv(os.path.join(self.traffic_path, f'{agent}_demand.csv'))
        
        self.period_t = 0

    
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
        self.period_t = 0
        # 1. Handle seeding for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # If using torch in your env setup:
            # torch.manual_seed(seed)
            
        # 2. Reset the clock to the initial discrete time slot
        #self.current_time = 0
        #logger.debug(f"[reset] current_time set to {self.current_time}")
        
        # 3. Restore all physical resource capacities back to maximum capacity
        for res in self.resources.values():
            res.reset()
        logger.debug(f"[reset] reset resources: {[(r.resource_id, r.available_capacity) for r in self.resources.values()]}")
            
        # 4. Clear out active task queues and wipe out the asynchronous ledger history
        for agent in self.agents:
            self.slices[agent].task_queue.clear()
        logger.debug(f"[reset] cleared task queues for all agents")
            
        #self._ready_rewards_ledger.clear()
        #logger.debug(f"[reset] cleared rewards ledger")
        
        #self.recorders = {agent: Recorder(self.slices[agent]) for agent in self.agents}
        

        
        # 5. Build and return initial observations and info mappings
        #print(f'--- Reset {self.current_time} ---')
        
        #if self.current_time > 0:
        #    self.current_time += 1

        observations = {agent: self._get_obs(agent)[1] for agent in self.agents}
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
        
        QUEUE_OVERFLOW = False
        
        # Initialize workload accumulation maps for all tracking resources in this slice
        accumulated_demands = {res_id: 0.0 for res_id in slice_obj.idx_to_resource}
        
        if self.test_demand is not None:
            new_task = Task(arrival_time=self.current_time, resource_demand={
                res_id: (self.test_demand * self.resources[res_id].capacity) / len(self.agents)
                for res_id in slice_obj.idx_to_resource
            })
            
        else:
            new_task = Task(arrival_time=self.current_time, resource_demand={
                res_id: self.demand[agent].iloc[self.current_time][res_id] / len(self.agents)
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
        
        self.current_queue[agent] = {}

        # Construct the features vector following a predictable indexing sequence
        obs_features = []
        for res_id in slice_obj.idx_to_resource:
            m = accumulated_demands[res_id] / self.resources[res_id].capacity

            obs_features.append(m)  # Normalize backlog by capacity for better learning stability
            self.current_queue[agent][res_id] = m

        
        for res_id in slice_obj.idx_to_resource:
            m = float(self.resources[res_id].available_capacity) / self.resources[res_id].capacity
            obs_features.append(m)  # Normalize available capacity by maximum capacity
        
        
        # Append the slice priority preferences (Equation 15 & 16 weights)
        obs_features.append(float(slice_obj.lambda_pref))  # Latency priority
        obs_features.append(float(slice_obj.rho_pref))     # Energy priority
        
        logger.debug(f"[_get_obs] agent={agent}, obs_features={obs_features}")
        # Convert to a flat float32 array suitable for standard neural network ingestion
        
        for res_id in slice_obj.idx_to_resource:
            if accumulated_demands[res_id] > self.max_queue_length * self.resources[res_id].capacity:
                QUEUE_OVERFLOW = True
                logger.warning(f"[_get_obs] Queue overflow detected for agent={agent}, resource={res_id}, accumulated_demand={accumulated_demands[res_id]}, capacity={self.resources[res_id].capacity}")
                
        return QUEUE_OVERFLOW,np.array(obs_features, dtype=np.float32)

        
    def step(self, actions):
        logger.debug(f"[step] current_time={self.current_time}, actions={actions}")
        
        self.previous_action = self.current_action
        self.current_action = []
        
        queue_now = copy.deepcopy(self.current_queue)
        systemic_energy = {agent: None for agent in self.agents}
        systemic_latency = {agent: None for agent in self.agents}
        
        items = list(actions.items())
        random.shuffle(items)
        
        if TEST_MODE:
            rewards = {agent: 0.0 for agent in self.agents}

                
        for res in self.resources.values():
            res.reset()
            
        # 1. Gather all requested allocations to enforce GLOBAL constraints (Eq 11)
        requested_allocations = {res_id: 0.0 for res_id in self.resources}
        agent_requests = {agent: {} for agent in self.agents}
        
        for agent, action in items:
            #print(agent)
            slice_obj = self.slices[agent]
            recorder = self.recorders[agent]
            
            for idx, res_id in enumerate(slice_obj.idx_to_resource):
                resource = self.resources[res_id]
                #normalized_action = (action[idx] + 1.0) / 2.0 
                #normalized_action = 0.3 * action[idx] + 0.7
                normalized_action = action[idx]*0.475 + 0.525
                #print(f'action: {normalized_action}')
                
                self.current_action.append(normalized_action)
                
                req = normalized_action * slice_obj.resource[res_id].capacity * self.resource_scaling_factor
                recorder.add_action(res_id, normalized_action)
                
                if req > resource.available_capacity:
                    #print(f"I AM HERE {agent}")
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

        deltas = {agent: {} for agent in self.agents}
        
        # 4. Schedule and apply proportional energy
        for agent in self.agents:
            slice_obj = self.slices[agent]
            recorder = self.recorders[agent]
            
            for res_id, allocated_amount in actual_allocations[agent].items():
                print(f'current_time {self.current_time}, res_id {res_id}')
                delta = self.scheduler.schedule(
                    task_queue=slice_obj.task_queue, 
                    res_id=res_id, 
                    allocated_amount=allocated_amount, 
                    total_resource_allocation=total_actual_allocations[res_id],
                    server_power=resource_powers[res_id],
                    current_time=self.current_time
                )
                
                deltas[agent][res_id] = delta
                
                recorder.add_active_time(res_id, 1 - delta)
            
            
            #systemic_energy[agent] = np.mean([(426.8324 - resource_powers[res_id] * (agent_requests[agent][res_id] / total_actual_allocations[res_id]) ) / 200    for res_id in slice_obj.idx_to_resource[:5]])
            
            x = np.array([(  MAX_QUEUE_LENGTH - max(self.current_queue[agent][res_id] - (agent_requests[agent][res_id] / resource.capacity), 0)  ) / (MAX_QUEUE_LENGTH ) for res_id, resource in self.resources.items()])
            #print([res_id for res_id, resource in self.resources.items()])
            
            #systemic_latency[agent] =  len(x) / np.sum(1.0 / x)
            systemic_latency[agent] =  1/2 * np.mean(x[:5]) + 1/2*np.mean(x[5:])
            #systemic_latency[agent] = np.mean(x) 


            
            #alloc = actual_allocations[agent]
            #current_q = self.current_queue[agent]
            
            #print(alloc)
            #print(current_q)
            
            #diff = np.array([(alloc[res_id] / self.resources[res_id].capacity) - current_q[res_id] for res_id in slice_obj.idx_to_resource])
            
            #queue_now = np.array([current_q[res_id] for res_id in slice_obj.idx_to_resource])
            
            #queue_control = np.max(diff ** 2 - queue_now ** 2)

            
            #systemic_latency = len(slice_obj.task_queue)
            
            
            #print(f'energy: {systemic_energy} - latency: {systemic_latency}')
            
            #lambda_i, rho_i = slice_obj.lambda_pref, slice_obj.rho_pref
            
            #rewards[agent] = (lambda_i *  (self.alpha / systemic_latency))  + (rho_i * (self.beta / systemic_energy)) - 10 * queue_control
            #rewards[agent] = - (lambda_i *  ( systemic_latency / self.alpha))  - (rho_i * (systemic_energy / self.beta)) - queue_control
            
                            
            # 5. Check for Completions & Rewards
            completed_tasks = []
            for task in slice_obj.task_queue:
                if task.is_complete():
                    completed_tasks.append(task)
                    
                    end_to_end_latency = max(list(task.completion_times.values()))
                    lambda_i, rho_i = slice_obj.lambda_pref, slice_obj.rho_pref
                    
                    reward_task = (lambda_i * (self.alpha / end_to_end_latency)) + \
                                  (rho_i * (self.beta / max(1e-5, task.accumulated_energy)))
                    
                    recorder.add_latency(task.arrival_time, end_to_end_latency)
                    #recorder.add_energy(task.arrival_time, float(task.accumulated_energy))
                    #recorder.add_reward(reward_task)

                    
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
                
        logger.debug(f"[step] incremented current_time to {self.current_time}")
        
        #if not TEST_MODE:
        #    rewards = {agent: 0.0 for agent in self.agents}
        
        self.current_time += 1
        self.period_t += 1

        observations = {}
        overflow_detected = False
        for agent in self.agents:
            overflow, obs = self._get_obs(agent)
            observations[agent] = obs
            if overflow:
                overflow_detected = True
                logger.warning(f"[step] Queue overflow detected for agent={agent} at time={self.current_time}")
        
        queue_next = self.current_queue
        
        #rewards = {agent: 0.0 for agent in self.agents}
        
        r = []
        
        e = {}
        for res_id, resource in self.resources.items():
            energy = 0
            if 'mec' in res_id:
                active_times = 1 - np.array([deltas[agent][res_id] for agent in self.agents])
                order = np.argsort(active_times)
                
                partial_usage = [actual_allocations[agent][res_id] / resource.capacity for agent in self.agents]
                total_usage = np.sum(partial_usage)
                
                start = None
                
                
                for o in order:
                    if start is None:
                        start = 0 
                    energy += resource.power_function(total_usage) * (active_times[o] - start)
                    
                    start = active_times[o]
                    total_usage -= partial_usage[o]
                
                if max(active_times) < 1:
                    energy += resource.power_function(0) * (1 - max(active_times))
                
                e[res_id] = energy
        
        
        for i, agent in enumerate(self.agents):
            agent_e = []
            recorder = self.recorders[agent]

            
            for res_id, resource in self.resources.items():
                if 'mec' in res_id:
                    active_times = 1 - np.array([deltas[agent][res_id] for agent in self.agents])
                    
                    partial_usage = [actual_allocations[agent][res_id] / resource.capacity for agent in self.agents]
                    total_usage = np.sum(partial_usage)
                    
                    nominator = partial_usage[i] * active_times[i]
                    denominator = np.sum([partial_usage[i] * active_times[i] for i in range(len(self.agents))])
                    coeff = nominator / denominator
                    

                    
                    agent_e.append((426.8324 - e[res_id] * coeff) / 200)
            
            systemic_energy[agent] = np.mean(agent_e)
            
            recorder.add_energy(systemic_energy[agent])
        
        
                    
                    
                
            

        for agent in self.agents:
            energy = systemic_energy[agent]
            latency = systemic_latency[agent]
            
            slice_obj = self.slices[agent]
            lambda_i, rho_i = slice_obj.lambda_pref, slice_obj.rho_pref


            
            queue_t = np.array([queue_now[agent][res_id] for res_id in self.slices[agent].idx_to_resource])
            queue_t_next = np.array([queue_next[agent][res_id] for res_id in self.slices[agent].idx_to_resource])
            #print(queue_t)
            #print(queue_t_next)
            #print(queue_t)
            #queue_control =  np.max(queue_t_next ** 2 - queue_t **2)
            
            queue_control = np.max(queue_t)  - 0.99 *np.max(queue_t_next)
            #print(f'queue_control {queue_control}')
            #rewards[agent] = - (lambda_i *  ( latency / self.alpha))  - (rho_i * (energy / self.beta)) - queue_control
            #rewards[agent] = (lambda_i *  (self.alpha / latency))  + (rho_i * (self.beta / energy)) -   queue_control
            
            action_control = 0
            
            if self.current_time > 2:
                a_t_prev = np.array(self.previous_action)
                a_t = np.array(self.current_action)
                
                diff = a_t - a_t_prev
                
                action_control = 0.1 * np.sum(diff**2)
            
            
            value = (lambda_i *  latency)  + (rho_i * energy) +  queue_control -  0*action_control
            print(f'latency: {latency} - energy: {energy} - queue_control : {queue_control} - action_contrl : {action_control}')
            r.append(value)
        #print(r)
        average_reward = np.mean(r)
        
        for agent in self.agents:
            self.recorders[agent].add_real_reward(average_reward)
        
        rewards = {agent: average_reward for agent in self.agents}
        #rewards = {agent: x for x in r}



            
        
        if overflow_detected:
            #rewards = {agent: PENALTY_REWARD for agent in self.agents}
            print('OVERFLOW')
            for agent in self.agents:
                recorder = self.recorders[agent]
                recorder.add_episode_end(self.current_time)
                recorder.add_overflow(self.current_time)
            
            
            logger.warning(f"[step] Applying penalty reward={PENALTY_REWARD} to all agents due to queue overflow at time={self.current_time}")
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {'overflow': True} for agent in self.agents}
            
            #for time_step, rewards in self._ready_rewards_ledger.items():
            #    if len(rewards) < len(self.agents):
            #        for agent in self.agents:
            #            if agent not in rewards:
            #                self._ready_rewards_ledger[time_step][agent] = PENALTY_REWARD
            
            for agent in self.agents:
                slice_obj = self.slices[agent]
                recorder = self.recorders[agent]

                for task in slice_obj.task_queue:
                    t_prime = task.arrival_time
                    end_to_end_latency = self.current_time - task.arrival_time + 1
                    if t_prime < self.current_time:
                        recorder.add_latency(task.arrival_time, end_to_end_latency)
                        #recorder.add_energy(task.arrival_time, float(task.accumulated_energy))
                    
                    
                    if t_prime in self._ready_rewards_ledger and agent not in self._ready_rewards_ledger[t_prime] and t_prime < self.current_time:
                        self._ready_rewards_ledger[t_prime][agent] = self.penalty_reward
                    else:
                        if t_prime not in self._ready_rewards_ledger and t_prime < self.current_time:
                            self._ready_rewards_ledger[t_prime] = {}
                            self._ready_rewards_ledger[t_prime][agent] = self.penalty_reward
                
                
                rewards[agent] = rewards[agent] + self.penalty_reward 
                self.recorders[agent].real_reward[-1] = self.recorders[agent].real_reward[-1] + self.penalty_reward
                
            print(f'time {self.current_time}- reward {rewards[agent]}')

            
            return observations, rewards, terminations, truncations, infos
        
        truncations = {agent: False for agent in self.agents}

        if self.period_t % 1000 == 0:
            for agent in self.agents:
                recorder = self.recorders[agent]
                recorder.add_episode_end(self.current_time)            
            
            truncations = {agent: True for agent in self.agents}
        
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        
        
        #logger.debug(f"[step] rewards={rewards}, terminations={terminations}")

        #logger.debug(f"[step] observations computed, returning step results")
        print(f'time {self.current_time}- reward {rewards[agent]}')

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
        
        #print(f"[get_ready_reward] Checking rewards ledger at time={self.current_time}, ledger={self._ready_rewards_ledger.keys()}")
        

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
        #print(f'self.current_time: {self.current_time}')
        
        #print(f"[get_ready_reward] Returning ready rewards for time steps: {list(ready.keys())}")
        #print(f"the reamining ledger time steps after flush: {list(self._ready_rewards_ledger.keys())}")
        return ready

    def save_statistics(self, output_dir=None):
        """Save collected statistics using per-slice Recorder objects."""
        print("[save_statistics] Saving recorder statistics...")
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
        self.energy = []
        self.reward = {}
        self.rejection = {resource: [] for resource in self.idx_to_resource}
        self.accumulated = {resource: [] for resource in self.idx_to_resource}
        self.episode_end = []
        self.real_reward = []
        self.overflow = []
        self.active_time = {resource: [] for resource in self.idx_to_resource}
    
    def add_active_time(self, id, active_time):
        self.active_time[id].append(active_time)
        
    def add_overflow(self, current_time):
        self.overflow.append(current_time)

    def add_real_reward(self, reward):
        self.real_reward.append(reward)
    
    def add_episode_end(self, current_time):
        self.episode_end.append(current_time)

    def add_accumulated(self, id, accumulated):
        self.accumulated[id].append(accumulated)

    def add_action(self, id, action):
        self.action[id].append(action)

    def add_allocation(self, id, allocation):
        self.allocation[id].append(allocation)

    def add_latency(self,t_prime, latency):
        self.latency[t_prime] = latency
    
    def add_energy(self, energy):
        self.energy.append(energy)
    
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
            #energy = {'energy': list(self.energy.values())}
            episode_end = {'episode_end': self.episode_end}
            overflow= {'episode_end': self.overflow}
            energy = {'energy': self.energy}

            
            #print(f"[Recorder.save_result] reward={reward}")
            data_to_save = {
                'action.csv': self.action,
                'allocation.csv': self.allocation,
                'latency.csv': latency,
                'energy.csv': energy,
                'rejection.csv': self.rejection,
                'reward.csv': reward,
                'accumulated.csv': self.accumulated,
                'episode_end.csv': episode_end,
                'overflow.csv': overflow,
                'active_time.csv': self.active_time
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
            
        return  226.8324 + 200 * utilization * utilization
        #return 426 * utilization * utilization * utilization

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
    
    def set_preferences(self, lambda_pref, rho_pref):
        self.lambda_pref = lambda_pref
        self.rho_pref = rho_pref
        logger.debug(f"[Slice.set_preferences] slice_id={self.slice_id}, lambda_pref={lambda_pref}, rho_pref={rho_pref}")

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
        return all(demand <= 1e-5 for demand in self.resource_demand.values())

from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self, task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time, delta = 1):
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
        resource_demand_of_tasks = [t.resource_demand.get(res_id) for t in active_tasks] 
        
        total = np.sum(resource_demand_of_tasks)
        partial = [r / total for r in resource_demand_of_tasks]
        
        logger.debug(f"[ProcessorSharingScheduler.schedule] res_id={res_id}, allocated_amount={allocated_amount}, active_tasks={len(active_tasks)}, current_time={current_time}")
        
        if not active_tasks or allocated_amount <= 0:
            logger.debug(f"[ProcessorSharingScheduler.schedule] no active tasks or no allocation, returning")
            return
            
        #share = allocated_amount / len(active_tasks) # This is g_{imt'}(k) for each task
        #logger.debug(f"[ProcessorSharingScheduler.schedule] share per task={share}")
        
        for i,task in enumerate(active_tasks):
            share = allocated_amount * partial[i]
            deduction = min(task.resource_demand[res_id], share)
            task.resource_demand[res_id] -= deduction
            
            logger.debug(f"[ProcessorSharingScheduler.schedule] task_arrival_time={task.arrival_time}, deduction={deduction}, remaining_demand={task.resource_demand[res_id]}")
            
            # --- EXACT ENERGY CALCULATION (Eq 9 & 10) ---
            if total_resource_allocation > 0:
                omega = share / total_resource_allocation
                task.accumulated_energy += omega * server_power
                logger.debug(f"[ProcessorSharingScheduler.schedule] omega={omega}, accumulated_energy={task.accumulated_energy}")
            
            if task.resource_demand[res_id] == 0 and res_id not in task.completion_times:
                task.completion_times[res_id] = current_time
                logger.debug(f"[ProcessorSharingScheduler.schedule] task completed for res_id={res_id} at time={current_time}")


class FairShareScheduler(BaseScheduler):
    def schedule(self, task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time, delta = 1):

        active_tasks_on_this_component = [t for t in task_queue if t.resource_demand.get(res_id, 0) > 1e-5]
        n_sub_tasks = len(active_tasks_on_this_component)
        
        #print(f'n_sub_tasks {n_sub_tasks}')
        
        if (delta <= 1e-5) or (n_sub_tasks == 0):
            return delta
        
        share = allocated_amount / n_sub_tasks
        #print(f'share {share}')
        
        t_finish = [t.resource_demand.get(res_id,0) / share for t in active_tasks_on_this_component]
        
        t_min_finish = min(min(t_finish), delta)
        
        #print(f't_min_finish {t_min_finish}, delta = {delta}')
        
                
        for t in active_tasks_on_this_component:
            t.resource_demand[res_id] = max(0, t.resource_demand[res_id] - t_min_finish * share)
            
            if (t.resource_demand[res_id] <= 1e-5) and (res_id not in t.completion_times):
                t.completion_times[res_id] = current_time +  t_min_finish
        
        delta = delta - t_min_finish
        
        return self.schedule(task_queue, res_id, allocated_amount, total_resource_allocation, server_power, current_time, delta)