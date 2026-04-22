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

mec2links = {
    1: [0, 1, 2, 3],
    2: [0, 4, 5, 6],
    3: [1, 4, 7, 8],
    4: [2, 5, 7, 9],
    5: [3, 6, 8, 9]}

class NetworkEnv(ParallelEnv):
    metadata = {"name": "network_env_v0"}

    def __init__(self, config_path=None, **kwargs):
        super().__init__(**kwargs)
        
        self._read_config(config_path)
        
        self.r_cpu = None # Available CPU resources at each MEC server
        self.r_bandwidth = None # Available bandwidth on each link
        self.cpu_req = None # CPU demand for each agent
        self.workload_size = None # Workload size for each agent
        self.allocated_cpu = None  # Allocated CPU resources for each agent
        self.allocated_bandwidth = None # Allocated bandwidth for each agent
        self.all_prefs = None # Concatenated preferences for all agents
        self.all_cpu_req = None # Total CPU demand across all agents
        self.all_workload_size = None # Total workload size across all agents
        

        self.time_step = 0
        self.checker_dict = {}
        self.window = 5
        self.recent_latencies = deque(maxlen = self.window)
        self.recent_energies = deque(maxlen = self.window)
    
    def _read_config(self, config_path):
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.cpu_capacity = np.array(config['server']['resource']['cpu'], dtype=np.float32)
        self.bw_capacity = np.array(config['link']['resource']['bandwidth'], dtype=np.float32)
        
        self.num_mecs = len(config['server']['resource']['cpu'])
        self.num_links = len(config['link']['resource']['bandwidth'])
        
        #self.num_agents = len(config['agent']['id'])
        agent_ids = config['agent']['id']
        self.agents = [f"agent_{i}" for i in agent_ids]
        self.possible_agents = self.agents[:]
        
        self.sim_workload = config['workload']
        self.sim_cpu_demand = config['cpu_demand']
        
        self.latency_pref = {agent: config['latency_preference'] for agent in self.agents}
        self.energy_pref = {agent: config['energy_preference'] for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # o_i = [rem_cpu, rem_bw, req_cpu, workload_size, pref_lambda, pref_rho]
        return Box(low=0, high=1, shape=((self.num_mecs + self.num_links) * 2 + 2,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # a_i = [cpu usage adjustment, bw usage adjustment] 
        return Box(low=0, high=1, shape=(self.num_mecs + self.num_links,), dtype=np.float32)

    def state(self):

        return np.concatenate([
            self.r_cpu / self.cpu_capacity,
            self.r_bandwidth / self.bw_capacity,
            self.all_cpu_req, 
            self.all_workload_size,
            self.all_prefs 
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.r_cpu = self.cpu_capacity.copy()
        self.r_bandwidth = self.bw_capacity.copy()
        
        # Initialize internal tracking for state()
        self.cpu_req = {agent: np.random.uniform(self.sim_cpu_demand['min'], self.sim_cpu_demand['max'], size=self.num_mecs).astype(np.float32) for agent in self.agents}
        self.workload_size = {agent: np.random.uniform(self.sim_workload['min'], self.sim_workload['max'], size=self.num_links).astype(np.float32) for agent in self.agents}
        self.all_cpu_req = np.mean(list(self.cpu_req.values()), axis=0)
        self.all_workload_size = np.mean(list(self.workload_size.values()), axis=0)
        self.all_prefs = np.concatenate([list(self.latency_pref.values()), list(self.energy_pref.values())])
        
        # For reward calculation and state tracking
        self.allocated_cpu = {agent : {} for agent in self.agents}
        self.allocated_bandwidth = {agent : {} for agent in self.agents}

        obs = {a: np.concatenate([self.r_cpu / self.cpu_capacity, self.r_bandwidth / self.bw_capacity, (self.cpu_req[a] - self.sim_cpu_demand['min']) / (self.sim_cpu_demand['max'] - self.sim_cpu_demand['min']), (self.workload_size[a] - self.sim_workload['min']) / (self.sim_workload['max'] - self.sim_workload['min']), [self.latency_pref[a], self.energy_pref[a]]])  for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        
        global_reward = {}
        some_reward_arrived = False
        
        
        #TODO: Add logic to handle invalid actions (e.g., requesting more resources than available) and calculate penalties if needed
        
        checker = Checker(agents=self.agents, num_mecs=self.num_mecs, time_step=self.time_step)
        
        
        # Convert items to a list so they can be shuffled
        items = list(actions.items())
        random.shuffle(items)
        
        #print(actions)
        
        w_b ={}
        w_m_b = {}
        latency = {}
        # perform resource allocation based on actions
        for agent_id, act in items:
            
            # Update allocated resources based on action
            allocated_cpu = act[:self.num_mecs] * self.cpu_capacity
            allocated_bandwidth = act[self.num_mecs:] * self.bw_capacity
            
            for i in range(self.num_mecs):
                if allocated_cpu[i] > self.r_cpu[i]:
                    allocated_cpu[i] = max(0,self.r_cpu[i] / self.cpu_capacity[i] - 0.05) * self.cpu_capacity[i]  # Cap allocation to available resources
                else:
                    allocated_cpu[i] = min(allocated_cpu[i], self.r_cpu[i] / self.cpu_capacity[i] - 0.05) * self.cpu_capacity[i]  # Ensure allocation does not exceed available resources
                    
                self.r_cpu[i] -= allocated_cpu[i]
                #self.allocated_cpu[agent_id].setdefault(self.time_step, []).append(allocated_cpu[i])
            
            
            for j in range(self.num_links):
                if allocated_bandwidth[j] > self.r_bandwidth[j]:
                    allocated_bandwidth[j] = max(0,self.r_bandwidth[j] / self.bw_capacity[j] - 0.05) * self.bw_capacity[j]  # Cap allocation to available resources
                else:
                    allocated_bandwidth[j] = min(allocated_bandwidth[j], self.r_bandwidth[j] / self.bw_capacity[j] - 0.05) * self.bw_capacity[j]  # Ensure allocation does not exceed available resources
                
                self.r_bandwidth[j] -= allocated_bandwidth[j]
                #self.allocated_bandwidth[agent_id].setdefault(self.time_step, []).append(allocated_bandwidth[j])
            
            checker.set_allocations(agent_id, allocated_cpu, allocated_bandwidth)
            
            # Calculate latency of each slice
            
            # latency associated with data transmission
            data_rate = self._calculate_datarate(signal_to_noise_ratio=40, allocated_bandwidth=allocated_bandwidth)
            w_a = np.sum(self.workload_size[agent_id] / (data_rate + 1e-6))  # Adding small value to avoid division by zero
            
            # latency associated with computation
            w_m_b[agent_id] = self.cpu_req[agent_id] / (allocated_cpu + 1e-6)  # Adding small value to avoid division by zero
            w_b[agent_id] = np.max(w_m_b[agent_id])  # Assuming latency is determined by the slowest MEC server processing the workload
            
            # Total latency  
            latency[agent_id] = w_a + w_b[agent_id]
        
        U_m = 1 - self.r_cpu / self.cpu_capacity
        
        f_Um_t = self._calculate_power_consumption(U_m)
        
        counter = {agent: np.ceil(w_m_b[agent]).astype(np.int32) for agent in self.agents}
        energy = {agent: f_Um_t for agent in self.agents}
        
        checker.set_counter(self.agents, self.num_mecs, counter)
        checker.set_energy(self.agents, self.num_mecs, energy)
        checker.set_latency(self.agents, latency)
        
        self.checker_dict[self.time_step] = checker
        
        #print(self.allocated_cpu)
        
        for i in self.checker_dict.keys():
            if self.checker_dict[i].update(self.agents, self.num_mecs, energy, r_cpu=self.r_cpu, r_bandwidth=self.r_bandwidth, allocated_cpu=self.allocated_cpu, allocated_bandwidth=self.allocated_bandwidth):
                
                some_reward_arrived = True
                
                
                self.recent_latencies.append(self.checker_dict[i].get_latency())
                self.recent_energies.append(self.checker_dict[i].get_energy())
                
                if len(self.recent_latencies) < self.window:
                    global_reward[i] = self.checker_dict[i].get_reward(min_latency=1, min_energy=1, latency_weight=self.latency_pref, energy_weight=self.energy_pref)
                else:
                    global_reward[i] = self.checker_dict[i].get_reward(min_latency=np.mean(self.recent_latencies), min_energy=np.mean(self.recent_energies), latency_weight=self.latency_pref, energy_weight=self.energy_pref)
                del self.checker_dict[i]
        
        
        # Update state for next step    
        self.cpu_req = {agent: np.random.uniform(self.sim_cpu_demand['min'], self.sim_cpu_demand['max'], size=self.num_mecs).astype(np.float32) for agent in self.agents}
        self.workload_size = {agent: np.random.uniform(self.sim_workload['min'], self.sim_workload['max'], size=self.num_links).astype(np.float32) for agent in self.agents}

        

        #Shared Reward 
        rewards = {a: -1 for a in self.agents}
        
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = {a: np.concatenate([self.r_cpu / self.cpu_capacity, self.r_bandwidth / self.bw_capacity, (self.cpu_req[a] - self.sim_cpu_demand['min']) / (self.sim_cpu_demand['max'] - self.sim_cpu_demand['min']), (self.workload_size[a] - self.sim_workload['min']) / (self.sim_workload['max'] - self.sim_workload['min']), [self.latency_pref[a], self.energy_pref[a]]])  for a in self.agents}
        infos = {a: {"update_reward": some_reward_arrived, "shared_reward": global_reward} for a in self.agents}
        
        #print(obs)
        self.time_step += 1

        return obs, rewards, terminations, truncations, infos


    def _calculate_power_consumption(self, resource_utilization_ratio):
        
        return 43.4779 * np.log(100 * resource_utilization_ratio) + 226.8324
    
    def _calculate_datarate(self, signal_to_noise_ratio, allocated_bandwidth):
        
        return allocated_bandwidth * np.log2(1 + signal_to_noise_ratio)
        
    
    
class Checker():
    def __init__(self, agents, num_mecs, time_step):
        self.energy = {agent: np.zeros(num_mecs, dtype=np.float32) for agent in agents}
        self.counter = {agent: np.zeros(num_mecs, dtype=np.int32) for agent in agents}
        self.global_counter = 0
        self.latency = {agent: 0 for agent in agents}
        self.time_step = time_step
        self.allocated_cpu = {agent : {} for agent in agents}
        self.allocated_bandwidth = {agent : {} for agent in agents}
        
        
    def set_allocations(self, agent, allocated_cpu, allocated_bandwidth):
        self.allocated_cpu[agent] = allocated_cpu
        self.allocated_bandwidth[agent] = allocated_bandwidth
        
    def set_counter(self, agents, num_mecs, counter):
        for agent in agents:
            for j in range(num_mecs):
                self.counter[agent][j] = counter[agent][j]
        
        self.global_counter = np.max(np.concatenate(list(self.counter.values())))
    
    def set_energy(self, agents, num_mecs, energy):
        for agent in agents:
            for j in range(num_mecs):
                self.energy[agent][j] = energy[agent][j]
    
    def set_latency(self, agents, latency):
        for agent in agents:
            self.latency[agent] = latency[agent]
    
    def update(self, agents, num_mecs, energy, r_cpu=None, r_bandwidth=None, allocated_cpu=None, allocated_bandwidth=None):
        for agent in agents:
            for j in range(num_mecs):
                if self.counter[agent][j] > 0:
                    self.energy[agent][j] += energy[agent][j]
                    self.counter[agent][j] -= 1
                    self.global_counter -= 1
                    
                    if self.counter[agent][j] == 0:
                        
                        r_cpu[j] +=  allocated_cpu[agent][j]
                        
                        for link in mec2links[j+1]:
                            r_bandwidth[link] += allocated_bandwidth[agent][link]
        
                    
        if self.global_counter == 0:
            return True
        else:
            return False
    
    def get_energy(self):
        return np.mean([np.sum(self.energy[agent]) for agent in self.energy])
    
    def get_latency(self):
        return np.mean([self.latency[agent] for agent in self.latency])
    
    def get_reward(self, min_latency, min_energy, latency_weight, energy_weight):
        
        rewards = []
        
        for agent in self.energy:
            energy_consumption = np.sum(self.energy[agent])
            normalized_energy = min_energy / (energy_consumption + 1e-6)  # Adding small value to avoid division by zero
            normalized_latency = min_latency / (self.latency[agent] + 1e-6)  # Adding small value to avoid division by zero
            reward = latency_weight[agent] * normalized_latency + energy_weight[agent] * normalized_energy
            
            rewards.append(reward)
        
        return np.mean(rewards)
        
        
        

    
        
        
# To use with TorchRL:
# from torchrl.envs.libs.pettingzoo import PettingZooWrapper
# env = PettingZooWrapper(NetworkEnv("config.json"), use_mask=False)