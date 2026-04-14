import numpy as np
import random
import heapq
import csv

SIM_TIME = 100 # seconds

slices = ["eMBB", "uRLLC", "mMTC"]


params = {
        "eMBB": {"lambda": 8, "size_range": (5, 15)},  # lambda (tasks / second), size_range (MBits)
        "uRLLC": {"lambda": 15, "size_range": (0.5, 2)},
        "mMTC": {"lambda": 20, "size_range": (0.1, 0.5)},
        }

class Task:
    def __init__(self, slice_type, time):
        p = params[slice_type]
        self.slice = slice_type
        self.timestamp = time
        self.size = np.random.uniform(*p["size_range"])
        
    def __repr__(self):
        return f"{self.timestamp:.6f}s | {self.slice} | {self.size:.3f}MBits"
    
    def to_list(self):
        return [f"{self.timestamp:.6f}", self.slice, f"{self.size:.3f}"]


class Event:
    def __init__(self, time, slice_type):
        self.time = time
        self.slice = slice_type
    
    def __lt__(self, other):
        return self.time < other.time

def next_arrival(current_time, lam):
    return current_time + np.random.exponential(1/lam)



for i in range(1, 11):
    np.random.seed(i)
    random.seed(i)
    
    event_queue = []
    
    for s in slices:
        t0 = next_arrival(0, params[s]["lambda"])
        heapq.heappush(event_queue, Event(t0, s))
    
    all_tasks = []
        
    while event_queue:
        event = heapq.heappop(event_queue)
        t = event.time
        s = event.slice
        
        
        if event.time > SIM_TIME:
            break
        
        task = Task(event.slice, event.time)
        all_tasks.append(task)
                
        lam = params[s]["lambda"]

        if s == "mMTC" and (t % 10) < 0.1:
            lam = 200
        
        t_next = next_arrival(t, lam)
        heapq.heappush(event_queue, Event(t_next, s))
        

    
    
    # --- SAVE TO CSV ---
    filename = f"traffic/simulation_seed_{i}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "SliceType", "Size_MBits"]) # Header
        for task in all_tasks:
            writer.writerow(task.to_list())
            
    print(f"Seed {i} completed: {len(all_tasks)} tasks saved to {filename}")

    
    

