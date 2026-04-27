import json

import numpy as np
import random
import heapq
import csv
import pandas as pd
import os

NUM_STEPS = 500000
NUM_AGENTS = 1
SEED = 0

random.seed(SEED)
np.random.seed(SEED)

agents =[]

for i in range(NUM_AGENTS):
    agents.append(f"agent_{i}")
    

demand = {agent: [] for agent in agents}


with open('experiments/experiment_0/configs/resource_config.json', 'r') as f:
    config = json.load(f)


for t in range(NUM_STEPS):
    for agent in agents:
        resource_demand = {}
        for item in config:
            if item['type'] == 'mec':
                resource_id = item['type'] + '_' + str(item['id'])
                #resource_demand[resource_id] = np.random.uniform(1, 2)
                resource_demand[resource_id] = 1.5
            elif item['type'] == 'link':
                resource_id = item['type'] + '_' + str(item['id'])
                #resource_demand[resource_id] = np.random.uniform(2.5, 5)
                resource_demand[resource_id] = 3.75
        
        demand[agent].append(resource_demand)

for agent in agents:
    df = pd.DataFrame(demand[agent])
    df.to_csv (f'{agent}_demand.csv', index=False)