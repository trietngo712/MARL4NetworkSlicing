import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import numpy as np
from network_env.network_env_v2 import NetworkEnvV2, Task, Resource, Slice, FIFOScheduler, ProcessorSharingScheduler




# Assuming the classes from the previous step are imported:
# NetworkEnvV2, Task, Resource, Slice, FIFOScheduler, ProcessorSharingScheduler

def test_problem_1():
    """
    Replicates Section 5.1 (Problem 1) [cite: 97-109]
    1 slice, 1 MEC, 1 Link.
    Constant allocation of 4 CPU & 3 bandwidth.
    Task arrives at t=0 with demand (G=10, D=6).
    """
    print("--- Running Test: Problem 1 (FIFO Scheduling) ---")
    
    # 1. Setup Environment & Inject custom resources for the test
    env = NetworkEnvV2(n_slices=1, scheduler=FIFOScheduler())
    env.set_test_mode(True)  # Enable test mode for deterministic behavior and logging
    env.resources = {
        'mec_1': Resource('mec_1', capacity=4.0, resource_type='mec'),   # Set max capacity to match allocation of 4 [cite: 98]
        'link_1': Resource('link_1', capacity=3.0, resource_type='link')  # Set max capacity to match allocation of 3 [cite: 98]
    }
    
    # Update slice mapping
    slice_0 = env.slices['slice_0']
    slice_0.idx_to_resource = ['mec_1', 'link_1']
    slice_0.resource = env.resources
    
    # 2. Add Task Z_{i=1, t=0} [cite: 98]
    task_z0 = Task(arrival_time=0, resource_demand={'mec_1': 10, 'link_1': 6})
    slice_0.add_task(task_z0)
    
    # Action of 1.0 translates to allocating 100% of the resource capacity 
    # (which exactly matches the 4 CPU and 3 BW required by the example)
    action = {'slice_0': np.array([1.0, 1.0])}
    
    # --- Time t = 1 --- [cite: 100, 101]
    env.step(action)
    assert task_z0.resource_demand['mec_1'] == 6, f"Expected 6, got {task_z0.resource_demand['mec_1']}"
    assert task_z0.resource_demand['link_1'] == 3, f"Expected 3, got {task_z0.resource_demand['link_1']}"
    print("t=1: Demands are correctly G=6, D=3")
    
    # --- Time t = 2 --- [cite: 102, 104]
    env.step(action)
    assert task_z0.resource_demand['mec_1'] == 2, f"Expected 2, got {task_z0.resource_demand['mec_1']}"
    assert task_z0.resource_demand['link_1'] == 0, f"Expected 0, got {task_z0.resource_demand['link_1']}"
    assert 'link_1' in task_z0.completion_times, "Link should be marked complete"
    print("t=2: Demands are correctly G=2, D=0 (Link finished)")
    
    # --- Time t = 3 --- [cite: 103, 106]
    _, rewards, _, _, _ = env.step(action)
    assert task_z0.is_complete(), "Task should be completely finished"
    assert rewards['slice_0'] > 0, "Reward should be generated at t=3"
    print("t=3: Task fully completed and reward granted. Problem 1 Test Passed!\n")


def test_problem_2():
    """
    Replicates Section 5.2 (Problem 2) [cite: 110-122]
    1 slice, 2 MECs.
    Always allocate 6 CPU to m=1 and 6 CPU to m=2.
    Uses "Always split equally" (Processor Sharing) scheduling.
    """
    print("--- Running Test: Problem 2 (Processor Sharing) ---")
    
    # 1. Setup Environment with Processor Sharing Scheduler [cite: 112]
    env = NetworkEnvV2(n_slices=1, scheduler=ProcessorSharingScheduler())
    env.set_test_mode(True)  # Enable test mode for deterministic behavior and logging
    # Inject 2 MECs with capacity 6 [cite: 111, 116]
    env.resources = {
        'mec_1': Resource('mec_1', capacity=6.0, resource_type='mec'),
        'mec_2': Resource('mec_2', capacity=6.0, resource_type='mec') 
    }
    
    slice_0 = env.slices['slice_0']
    slice_0.idx_to_resource = ['mec_1', 'mec_2']
    slice_0.resource = env.resources
    
    # Always allocate maximum capacity (6 on both) [cite: 116]
    action = {'slice_0': np.array([1.0, 1.0])}
    
    # 2. Add Task Z_{i=1, t=0} [cite: 113]
    task_z0 = Task(arrival_time=0, resource_demand={'mec_1': 10, 'mec_2': 6})
    slice_0.add_task(task_z0)
    
    # --- Time t = 1 --- [cite: 119]
    env.step(action)
    assert task_z0.resource_demand['mec_1'] == 4, f"Expected 4, got {task_z0.resource_demand['mec_1']}" # [cite: 118]
    assert task_z0.resource_demand['mec_2'] == 0, f"Expected 0, got {task_z0.resource_demand['mec_2']}" # [cite: 118]
    print("t=1: Z_0 demands are G_m1=4, G_m2=0 (m2 finished for Z_0)")
    
    # 3. Add Task Z_{i=1, t=1} [cite: 114, 115]
    task_z1 = Task(arrival_time=1, resource_demand={'mec_1': 10, 'mec_2': 6})
    slice_0.add_task(task_z1)
    
    # --- Time t = 2 --- 
    env.step(action)
    
    # Processor sharing splits the 6 allocation on mec_1 equally between z0 and z1 (3 each) [cite: 121]
    assert task_z0.resource_demand['mec_1'] == 1, f"Expected Z0 G_m1=1, got {task_z0.resource_demand['mec_1']}" # 4 - 3 = 1 [cite: 121]
    assert task_z1.resource_demand['mec_1'] == 7, f"Expected Z1 G_m1=7, got {task_z1.resource_demand['mec_1']}" # 10 - 3 = 7 [cite: 121]
    
    # Z1 gets all 6 allocation on mec_2 since Z0 is already done with it [cite: 121]
    assert task_z1.resource_demand['mec_2'] == 0, f"Expected Z1 G_m2=0, got {task_z1.resource_demand['mec_2']}" # 6 - 6 = 0 [cite: 121]
    
    assert not task_z0.is_complete(), "Z0 should not be finished" # [cite: 122]
    assert not task_z1.is_complete(), "Z1 should not be finished" # [cite: 122]
    print("t=2: Resources split correctly. Neither task finished. Problem 2 Test Passed!\n")

if __name__ == "__main__":
    test_problem_1()
    test_problem_2()