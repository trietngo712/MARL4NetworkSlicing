import json

import numpy as np
import random
import heapq
import csv
import pandas as pd
import os

NUM_STEPS = 1000
NUM_AGENTS = 3
SEED = 1

random.seed(SEED)
np.random.seed(SEED)

agents =[]

for i in range(NUM_AGENTS):
    agents.append(f"slice_{i}")
    

demand = {agent: [] for agent in agents}


with open('experiments/experiment_1/configs/resource_config.json', 'r') as f:
    config = json.load(f)


for t in range(NUM_STEPS + 1):
    for agent in agents:
        resource_demand = {}
        for item in config:
            if item['type'] == 'mec':
                resource_id = item['type'] + '_' + str(item['id'])
                resource_demand[resource_id] = np.random.uniform(0.3, 0.7) * 4
                #resource_demand[resource_id] = 1
            elif item['type'] == 'link':
                resource_id = item['type'] + '_' + str(item['id'])
                resource_demand[resource_id] = np.random.uniform(0.3, 0.7) * 10
                #resource_demand[resource_id] = 1
        
        demand[agent].append(resource_demand)

for agent in agents:
    df = pd.DataFrame(demand[agent])
    df.to_csv (f'{agent}_demand.csv', index=False)