import functools
import pandas as pd
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

class NetworkEnv(ParallelEnv):
    metadata = {"name": "network_env_v3"}

    def __init__(self, csv_path, decision_interval=1.0):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df['Timestamp'] = self.df['Timestamp'].astype(float)
        
        self.decision_interval = decision_interval
        self.slices = ["eMBB", "uRLLC", "mMTC"]
        self.agents = self.slices
        self.possible_agents = self.agents[:]
        
        # --- Global Resource Constraints ---
        self.MAX_BANDWIDTH = 25.0  # Mbps
        self.MAX_COMPUTE = 10000.0  # 40,000 MHz (40 GHz)
        
        self.processing_density = {"eMBB": 40, "uRLLC": 120, "mMTC": 15}
        self.latency_targets = {"eMBB": 0.100, "uRLLC": 0.005, "mMTC": 0.500}

        # --- Spaces ---
        # Action: [Bandwidth Weight, Compute Weight]
        self.action_spaces = {
            a: spaces.Box(low=0.01, high=1.0, shape=(2,), dtype=np.float32) 
            for a in self.agents
        }
        
        # Observation: [Demand, Avg Size, Task Count, Queue, BW_Share_%, CPU_Share_%]
        self.observation_spaces = {
            a: spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32) 
            for a in self.agents
        }

        # --- Internal State ---
        self.pending_tasks = {s: [] for s in self.slices}
        self.current_time = 0.0
        # Track previous allocation for observations
        self.last_allocations = {a: np.zeros(2, dtype=np.float32) for a in self.agents}

    def reset(self, seed=None, options=None):
        self.current_time = 0.0
        self.agents = self.possible_agents[:]
        self.pending_tasks = {s: [] for s in self.slices}
        self.last_allocations = {a: np.zeros(2, dtype=np.float32) for a in self.agents}
        return self._get_observations(), {a: {} for a in self.agents}

    def _get_observations(self):
        end_time = self.current_time + self.decision_interval
        mask = (self.df['Timestamp'] >= self.current_time) & (self.df['Timestamp'] < end_time)
        interval_data = self.df[mask]
        
        obs = {}
        for s in self.slices:
            slice_data = interval_data[interval_data['SliceType'] == s]
            queue_backlog = sum([task[1] for task in self.pending_tasks[s]])
            
            # Local observation + Resource usage percentage
            obs[s] = np.array([
                slice_data['Size_MBits'].sum(),
                slice_data['Size_MBits'].mean() if len(slice_data) > 0 else 0,
                len(slice_data),
                queue_backlog,
                self.last_allocations[s][0], # BW % used in last step
                self.last_allocations[s][1]  # CPU % used in last step
            ], dtype=np.float32)
        return obs

    def state(self):
        """Returns the global state for Centralized Training."""
        all_obs = self._get_observations()
        # Concatenate all agent observations + current time info
        global_state = np.concatenate([all_obs[a] for a in self.possible_agents])
        return global_state.astype(np.float32)

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        
        #print(actions)
        
        # 1. Capture new arrivals
        end_time = self.current_time + self.decision_interval
        mask = (self.df['Timestamp'] >= self.current_time) & (self.df['Timestamp'] < end_time)
        new_arrivals = self.df[mask]
        for s in self.slices:
            slice_arrivals = new_arrivals[new_arrivals['SliceType'] == s]
            for _, row in slice_arrivals.iterrows():
                self.pending_tasks[s].append([row['Timestamp'], row['Size_MBits']])

        # 2. Resource Normalization
        total_bw_w = sum(actions[a][0] for a in self.agents)
        total_cpu_w = sum(actions[a][1] for a in self.agents)

        for s in self.slices:
            time = self.current_time
            # Calculate Shares
            bw_percent = actions[s][0] / total_bw_w
            cpu_percent = actions[s][1] / total_cpu_w
            
            # Store for next observation
            self.last_allocations[s] = np.array([bw_percent, cpu_percent], dtype=np.float32)

            bw_share = bw_percent * self.MAX_BANDWIDTH
            cpu_share = cpu_percent * self.MAX_COMPUTE
            
            # 3. Process Capacity (Bottleneck)
            #communication_cap = bw_share * self.decision_interval
            #computation_cap = (cpu_share * self.decision_interval) / self.processing_density[s]
            #effective_capacity = min(communication_cap, computation_cap)
            
            # 4. Queue Processing
            while self.pending_tasks[s]:
                #print(f'Current Time: {self.current_time:.2f}s, Processing {s} task with arrival at {self.pending_tasks[s][0][0]:.2f}s and size {self.pending_tasks[s][0][1]:.2f} Mbits')
                arrival_time, size = self.pending_tasks[s][0]
                
                time = max(time, arrival_time)  # Task can only start after it arrives
                
                processed_time = size / cpu_share * self.processing_density[s] + size / bw_share if cpu_share > 0 and bw_share > 0 else float('inf') 
                
                if time + processed_time <= end_time:
                    self.pending_tasks[s].pop(0)
                    latency = time + processed_time - arrival_time
                    if latency <= self.latency_targets[s]:
                        rewards[s] += 1.0
                    else:
                        rewards[s] -= (20.0 if s == "uRLLC" else 2.0)
                    
                    time += processed_time
                #else:
                #    self.pending_tasks[s][0][1] -= effective_capacity
                #    effective_capacity = 0
                else:
                    break

        self.current_time += self.decision_interval
        observations = self._get_observations()
        
        env_done = self.current_time >= self.df['Timestamp'].max()
        terminations = {a: env_done for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        
        mean_reward = np.mean(list(rewards.values()))
        rewards = {a: mean_reward for a in self.agents}

        if env_done: self.agents = []

        return observations, rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    

