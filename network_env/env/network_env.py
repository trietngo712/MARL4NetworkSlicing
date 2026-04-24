from collections import deque
import functools
import random
import numpy as np
import json
import torch
from gymnasium.spaces import Box, Dict
from pettingzoo import ParallelEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from collections import defaultdict
import pandas as pd
import os 

NUM_AGENTS = 4
WINDOW = 20

class NetworkEnv(ParallelEnv):
    def __init__(self, config_path = None):
        super().__init__()
        
        if config_path is None:
            raise ValueError("Config path must be provided.")
        
        self.config_path = config_path
        
                #Reset the environment to an initial state and return the initial observations and infos for all agents.
        self.time_step = 0
        
        #read config file and create resources and slices
        self._read_config(self.config_path)
        
        self.agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.possible_agents = self.agents[:]
        
        self.slices = {}
        for agent in self.agents:
            self.slices[agent] = Slice(id=agent, resources=self.resources)
        
        # We use deques to store recent energy consumption and latency for each agent, which will be used to calculate the minimum energy and latency for reward calculation.
        self.recent_energy = {agent: deque(maxlen=WINDOW) for agent in self.agents}
        self.recent_latency = {agent: deque(maxlen=WINDOW) for agent in self.agents}
        
        self.reward_track = {} #{time_step: {agent: reward}}
        
        self.ready_reward = {}
        self.is_ready_value = False

        

        
    def say_hello(self):
        print("Hello from NetworkEnv!")
    
    def _is_reward_ready(self, time_step):
        for _, reward in self.reward_track[time_step].items():
            if reward is None:
                return False
        return True

    def _minimum_latency(self):
        min_latency = 1
        condition = np.all([len(self.recent_latency[agent]) >= WINDOW for agent in self.agents])
        
        if condition:
            min_latency = np.mean([np.min(self.recent_latency[agent]) for agent in self.agents])
        
        return min_latency
    
    def _minimum_energy(self):
        min_energy = 100
        condition = np.all([len(self.recent_energy[agent]) >= WINDOW for agent in self.agents])
        
        if condition:
            min_energy = np.mean([np.min(self.recent_energy[agent]) for agent in self.agents])
        
        return min_energy

        
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        number_of_resources = self.slices[agent].number_of_resources()
        return Box(low=0.0, high=np.inf, shape=(number_of_resources * 2 + 2,), dtype=np.float32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        number_of_resources = self.slices[agent].number_of_resources()
        return Box(low=0.0, high=1.0, shape=(number_of_resources,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        print("-----------------RESET-------------------")
        #Reset the environment to an initial state and return the initial observations and infos for all agents.
        self.time_step = 0
        
        #read config file and create resources and slices
        self._read_config(self.config_path)
        
        self.slices = {}
        for agent in self.agents:
            self.slices[agent] = Slice(id=agent, resources=self.resources)
        
        # We use deques to store recent energy consumption and latency for each agent, which will be used to calculate the minimum energy and latency for reward calculation.
        self.recent_energy = {agent: deque(maxlen=WINDOW) for agent in self.agents}
        self.recent_latency = {agent: deque(maxlen=WINDOW) for agent in self.agents}
        
        self.reward_track = {} #{time_step: {agent: reward}}

        
        self.demand = {}
        for agent in self.agents:
            self.demand[agent] = pd.read_csv(os.path.join('traffic', f'{agent}_demand.csv'))
        
        self.rejection_counts = {agent: 0 for agent in self.agents}
        
        self.ready_reward = {}
        self.is_ready_value = False

        
 
        obs = self._get_observation(self.time_step)
        #print("Initial Observation:", obs)
        infos = {agent: {} for agent in self.agents}
        
        return obs, infos
    
    def step(self, actions):
        print(f'-----------------{self.time_step}-------------------')
        #print(self.reward_track)
        #print(f"TIME STEP: {self.time_step}")
        # We process agents in random order to avoid bias towards certain agents.
        items = list(actions.items())
        random.shuffle(items)
        reward = {agent: None for agent in self.agents}
        self.reward_track[self.time_step] = reward
        
        for agent, action in items:
            #This is the beginning of the time step
            #1. Get the demand for this time step
            #2. Create a register for this time 
            #3. For each resource, check if the demand can be met with the allocated resource. If yes, create a task and add to register. If no, reject the task.
            
            slice = self.slices[agent]
            demand = self.demand[agent].iloc[self.time_step]
            register = Register(creation_time=self.time_step)
            
            for idx, resource_id in enumerate(slice.idx_to_resource):
                resource_allocation = action[idx] * slice.get_resource_by_index(idx).capacity
                
                # We add a safety margin here
                # If the requested resource allocation exceeds the available resource, 
                # we adjust it to be slightly less than the available resource to avoid over-allocation.
                # However, if the available resource is very low (less than 5% of capacity),
                # we set the allocation to 0 to avoid creating tasks that can never be completed.
                if resource_allocation > slice.get_resource_by_index(idx).available:
                    adjusted_amount = slice.get_resource_by_index(idx).available - 0.05 * slice.get_resource_by_index(idx).capacity
                    resource_allocation = max(adjusted_amount, 0)
                    
                
                duration = demand[resource_id] / resource_allocation if resource_allocation != 0 else 0
                
                # We only accept the task if the duration is greater than 0 and less than the max latency requirement of the slice.
                if duration > 0 and duration <= slice.max_latency:
                    task = Task(resource_id, self.time_step, duration=duration, resource_allocation=resource_allocation)
                    register.add_task(task, slice.resources)
                else:
                    self.rejection_counts[agent] += 1
            
            # Only add register to slice if there is at least one task in the register.
            # If there is no task, it means all tasks are rejected
            # and we don't need to add an empty register to the slice.
            if register.number_of_tasks() > 0:
                slice.add_register(register)
            else:
                self.reward_track[self.time_step][agent] = 0 # If all tasks are rejected, we set reward to 0 for this time step for this agent.
            
            
            #This is the end of the time step
            #1. Update energy consumption of active tasks in the register
            #2. Release resources of tasks that would be done before next time step
            
            register_dict = slice.get_all_registers()
            for register in register_dict.values():
                #What happens here is 
                #1. the register call update function
                #2. update function will update energy consumption of active tasks
                #3. update function will release resources of tasks that would be done before next time step
                #print(f'agent: {agent}, time_step: {self.time_step}, register creation time: {register.creation_time}')
                register.update(slice.resources, self.time_step)
            
            #This is the beginning of the next time step, but before new demand
            #We calculate the reward if there is any register finished before this point
            reward = 0
            creation_times = []
            for creation_time, register in register_dict.items():
                if register.is_done(self.time_step + 1):
                    total_latency = register.get_total_latency()
                    total_energy = register.get_total_energy_consumption()
                    
                    self.recent_latency[agent].append(total_latency)
                    self.recent_energy[agent].append(total_energy)
                    
                    #print(f"Time Step: {self.time_step}, Agent: {agent}, Total Latency: {total_latency:.2f}, Total Energy: {total_energy:.2f}")
                    #print(f"{self._minimum_latency() / total_latency:.2f}, {self._minimum_energy() / total_energy:.2f}")
                    
                    a = self._minimum_latency() / total_latency if total_latency > 0 else 0
                    b = self._minimum_energy() / total_energy if total_energy > 0 else 0
                                        
                    reward = slice.latency_coeff * a + slice.energy_coeff * b   
                    
                    
                    #print(f'time step {self.time_step}')
                    #print(f'register creation time {register.creation_time}')
                
                    self.reward_track[register.creation_time][agent] = reward
                    creation_times.append(creation_time)
            
            for creation_time in creation_times:
                slice.remove_register(creation_time)

        
        reward_at_time = {}
        for time in self.reward_track.keys():
            # We check if the reward for this time step is ready. 
            # If it is, we calculate the average reward of all agents 
            # for this time step and assign it to all agents.
            if time in self.reward_track and self._is_reward_ready(time):
                for agent, reward in self.reward_track[time].items():
                    if reward is None:
                        raise ValueError(f"Reward for agent {agent} at time {time} is not ready.")
                
                avg_reward = np.mean(list(self.reward_track[time].values()))
                reward = {agent: avg_reward for agent in self.agents}
                reward_at_time[time] = reward
                
        if len(reward_at_time) > 0:
            #print(self.reward_track)
            for time in reward_at_time.keys():
                del self.reward_track[time] # We delete the reward from reward_track after using it to save memory.
            #print(self.reward_track)
                
        reward = {agent: -1 for agent in self.agents}
        
        if len(reward_at_time) > 0:
            self.is_ready_value = True
            self.ready_reward = reward_at_time
            
                
        self.time_step += 1 
          
        obs = self._get_observation(self.time_step)
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return obs, reward, terminations, truncations, infos
    
    def is_ready(self):
        return self.is_ready_value
    
    def get_ready_reward(self):
        if not self.is_ready:
            raise ValueError("Reward is not ready yet.")
        self.is_ready_value = False # Reset the ready state after getting the reward
        
        return self.ready_reward
        
    def state(self):
        return np.concatenate([self._get_observation(agent) for agent in self.agents])
        
    
    def _get_observation(self, time_step):
        obs = {}
        system_state = {agent: [] for agent in self.agents}
        demand_state = {agent: [] for agent in self.agents}
        preference_state = {agent: [] for agent in self.agents}
        
        for agent in self.agents:
            slice = self.slices[agent]
            
            for resource_id in slice.idx_to_resource:
                resource = slice.get_resource_by_id(resource_id)
                system_state[agent].append(resource.available)
                demand_state[agent].append(self.demand[agent].iloc[time_step][resource_id])
            
            preference_state[agent].append(slice.latency_coeff)
            preference_state[agent].append(slice.energy_coeff)
            obs[agent] = np.array(system_state[agent] + demand_state[agent] + preference_state[agent], dtype=np.float32)
            #print(len(obs[agent]), len(system_state), len(demand_state), len(preference_state))
        
        return obs
        
        
    def _read_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.resources = {}
            
        for i, item in enumerate(config):
            if item['type'] == 'mec':
                resource_id = item['type'] + '_' + str(item['id'])
                self.resources[resource_id] = Resource('cpu', item['resources']['cpu'], item['type'], item['id'])
                self.resources[resource_id].set_energy_calculator(EnergyCalculator())
            elif item['type'] == 'link':
                resource_id = item['type'] + '_' + str(item['id'])
                self.resources[resource_id] = Resource('bandwidth', item['resources']['bandwidth'], item['type'], item['id'])
                self.resources[resource_id].set_energy_calculator(EnergyCalculator())


class Resource():
    def __init__(self, resource_type, capacity, entity, entity_id):
        self.type = resource_type
        self.capacity = capacity
        self.available = capacity
        self.entity = entity
        self.entity_id = entity_id
        self.energy_calulator = None
    
    def set_energy_calculator(self, energy_calculator):
        self.energy_calulator = energy_calculator

    def allocate(self, amount):
        if amount > self.available:
            raise ValueError(f"Not enough {self.type} available to allocate. Requested: {amount}, Available: {self.available}")
        self.available -= amount
    
    def release(self, amount):
        self.available += amount
        if self.available > self.capacity:
            self.available = self.capacity  # Ensure we don't exceed capacity
    
    def get_energy_consumption(self):
        if self.energy_calulator is None:
            raise ValueError("Energy calculator not set for this resource.")
        return self.energy_calulator.calculate_energy(self.type,self.capacity, self.available)

class Slice():
    def __init__(self, id, resources, latency_coeff = 0.5, energy_coeff = 0.5, max_latency = 4):
        self.id = id
        self.resources = resources # {resource_id: Resource}
        self.latency_coeff = latency_coeff
        self.energy_coeff = energy_coeff
        self.max_latency = max_latency # measured in time steps
        self.registers = {}        
        
        self.idx_to_resource = []
        for resource_id in resources.keys():
            self.idx_to_resource.append(resource_id)
    
    def get_register(self, time):
        return self.registers[time]
    
    def get_all_registers(self):
        return self.registers
    
    def add_register(self, register):
        self.registers[register.creation_time] = register
        
    def number_of_resources(self):
        return len(self.resources)
    
    def get_resource_by_index(self, idx):
        resource_id = self.idx_to_resource[idx]
        return self.resources[resource_id]
    
    def get_resource_by_id(self, resource_id):
        return self.resources[resource_id]
    
    def remove_register(self, creation_time):
        #print('call remove register')
        #print(self.registers)
        del self.registers[creation_time]
        #print(self.registers)
    

class Task():
    def __init__(self, resource_id, start_from, duration, resource_allocation):
        self.resource_id = resource_id
        self.start_from = start_from
        self.duration = duration
        self.resource_allocation = resource_allocation
        self.end = start_from + duration
        self.energy_consumption = 0
        self.is_consumed = False
        self.is_released = False
        self.number_of_updates = np.ceil(duration)
    
    def get_task_id(self):
        return f"{self.resource_id}_{self.start_from:.2f}_{self.end:.2f}"
    
    def consume(self, resource):
        if self.is_consumed:
            raise ValueError("Task has already been consumed.")
        self.is_consumed = True
        resource[self.resource_id].allocate(self.resource_allocation)
    
    def release(self, resource):
        if self.is_released:
            raise ValueError("Task has already been released.")
        self.is_released = True
        resource[self.resource_id].release(self.resource_allocation)
    
    def update_energy_consumption(self, resource):
        #print(f'Updating energy consumption for task on resource {self.resource_id}. Remaining updates: {self.number_of_updates}. Duration: {self.duration:.2f}, Max_updates: {np.ceil(self.duration)}, end: {self.end:.2f}')

        if self.number_of_updates <= 0:
            raise ValueError("Task has already been updated for the required number of updates.")
        self.number_of_updates -= 1
        this_resource = resource[self.resource_id]
        self.energy_consumption += this_resource.get_energy_consumption() * self.resource_allocation / this_resource.capacity
    
    def get_energy_consumption(self):
        return self.energy_consumption
    
    def is_done(self, current_time):
        return current_time >= self.end
    
    def get_resource_id(self):
        return self.resource_id
    
    def get_resource_allocation(self):
        return self.resource_allocation
    
    def get_duration(self):
        return self.duration
    
class Register():
    def __init__(self, creation_time = None):
        #print(f'Creating register at time {creation_time}')
        self.tasks  = []
        self.active_tasks = []
        self.creation_time = creation_time
    
    def add_task(self, task, resource):
        task.consume(resource)
        self.tasks.append(task)
        
    def number_of_tasks(self):
        return len(self.tasks)
        
    def is_done(self, current_time):
        return np.all([task.is_done(current_time) for task in self.tasks])
    
    def _update_energy_consumption(self, resource, current_time):
        for task in self.active_tasks:
                task.update_energy_consumption(resource)
    
    def update(self, resource, current_time):
        #print(f'tasks: {[task.get_task_id() for task in self.tasks]}')
        #Get energy consumption of current active tasks
        self.active_tasks = [task for task in self.tasks if not task.is_done(current_time)]
        #print(f'active tasks: {[task.get_task_id() for task in self.active_tasks]}')
        self._update_energy_consumption(resource, current_time)
        
        #Release resources of tasks that would be done before netxt time step
        for task in self.active_tasks:
            if task.is_done(current_time + 1):
                task.release(resource)
        
    def get_total_energy_consumption(self):
        return np.sum([task.get_energy_consumption() for task in self.tasks])
    
    def get_total_latency(self):
        cpu_tasks = [task for task in self.tasks if 'mec' in task.get_resource_id()]
        bandwidth_tasks = [task for task in self.tasks if 'link' in task.get_resource_id()]
 
        
        compute_latency = np.max([task.get_duration() for task in cpu_tasks]) if cpu_tasks else 0
        bandwidth_latency = np.sum([task.get_duration() for task in bandwidth_tasks]) if bandwidth_tasks else 0
        
        return compute_latency + bandwidth_latency

class EnergyCalculator():
    def __init__(self):
        None
        
    def calculate_energy(self, resource_type, capacity, available):
        if resource_type == 'cpu':
            utilization = (capacity - available) / capacity
            return  43.4779 * np.log(100 * utilization) + 226.8324
        elif resource_type == 'bandwidth':
            return 0
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
    
        
    
    
