"""
Parse network_env debug logs and extract metrics to CSV for visualization.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_file_path=None, output_csv='extracted_metrics.csv'):
    """
    Parse debug log file and extract key metrics into CSV format.
    
    Args:
        log_file_path: Path to log file or directory containing 'network_env_debug.log'
                      If None, searches in current directory
    
    Extracts:
    - current_time: simulation time step
    - agent: slice agent name
    - action: actions taken by agent
    - reward: reward calculated
    - latency: end-to-end latency
    - energy: accumulated energy
    - allocation: resource allocation
    - deduction: actual resource processed
    """
    
    # Determine actual log file path
    if log_file_path is None:
        log_file_path = Path('network_env_debug.log')
    else:
        log_file_path = Path(log_file_path)
        # If directory given, look for log file inside it
        if log_file_path.is_dir():
            log_file_path = log_file_path / 'network_env_debug.log'
    
    metrics = []
    current_step = None
    step_data = defaultdict(lambda: defaultdict(dict))
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Extract timestamp and message
                if ' - ' not in line:
                    continue
                    
                # ===== STEP START =====
                if '[step]' in line and 'current_time=' in line:
                    match = re.search(r'current_time=(\d+)', line)
                    if match:
                        current_step = int(match.group(1))
                        step_data[current_step] = defaultdict(dict)
                
                # ===== ACTIONS =====
                if '[step]' in line and 'actions=' in line:
                    match = re.search(r'actions=({[^}]+})', line)
                    if match:
                        step_data[current_step]['actions'] = match.group(1)
                
                # ===== ALLOCATIONS =====
                if '[step]' in line and 'actual_allocations=' in line:
                    match = re.search(r'actual_allocations=({[^}]+})', line)
                    if match:
                        step_data[current_step]['allocations'] = match.group(1)
                
                # ===== REWARDS =====
                if 'reward_task=' in line:
                    agent_match = re.search(r'agent=(\w+)', line)
                    reward_match = re.search(r'reward_task=([0-9\.\-e]+)', line)
                    latency_match = re.search(r'end_to_end_latency=(\d+)', line)
                    energy_match = re.search(r'accumulated_energy=([0-9\.\-e]+)', line)
                    
                    if agent_match and reward_match:
                        agent = agent_match.group(1)
                        reward = float(reward_match.group(1))
                        latency = int(latency_match.group(1)) if latency_match else None
                        energy = float(energy_match.group(1)) if energy_match else None
                        
                        metrics.append({
                            'time_step': current_step,
                            'agent': agent,
                            'action': step_data[current_step].get('actions', ''),
                            'reward': reward,
                            'latency': latency,
                            'energy': energy,
                            'allocation': step_data[current_step].get('allocations', '')
                        })
                
                # ===== FIFO SCHEDULER DEDUCTIONS =====
                if '[FIFOScheduler.schedule]' in line and 'deduction=' in line:
                    deduction_match = re.search(r'deduction=([0-9\.\-e]+)', line)
                    energy_match = re.search(r'accumulated_energy=([0-9\.\-e]+)', line)
                    
                    if deduction_match and current_step is not None and metrics:
                        deduction = float(deduction_match.group(1))
                        # Add to last metric if on same step
                        if metrics[-1]['time_step'] == current_step:
                            metrics[-1]['deduction'] = deduction
        
        # Write to CSV
        if metrics:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['time_step', 'agent', 'reward', 'latency', 'energy', 'deduction', 'action', 'allocation']
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in metrics:
                    row = {key: metric.get(key, '') for key in fieldnames}
                    writer.writerow(row)
            
            print(f"✓ Extracted {len(metrics)} metrics to {output_path}")
            print(f"  Columns: {', '.join(fieldnames)}")
            return metrics, output_path
        else:
            print("✗ No metrics found in log file")
            return [], None
            
    except FileNotFoundError:
        print(f"✗ Log file not found: {log_file_path}")
        return [], None
    except Exception as e:
        print(f"✗ Error parsing log file: {e}")
        return [], None


def create_summary_stats(metrics_list, output_csv='metrics_summary.csv'):
    """
    Create summary statistics grouped by agent and time step.
    """
    if not metrics_list:
        print("No metrics to summarize")
        return
    
    summary = defaultdict(lambda: {
        'count': 0,
        'avg_reward': 0,
        'avg_latency': 0,
        'avg_energy': 0,
        'avg_deduction': 0
    })
    
    for metric in metrics_list:
        agent = metric.get('agent', 'unknown')
        key = agent
        
        summary[key]['count'] += 1
        summary[key]['avg_reward'] += metric.get('reward', 0)
        
        if metric.get('latency') is not None:
            summary[key]['avg_latency'] += metric['latency']
        if metric.get('energy') is not None:
            summary[key]['avg_energy'] += metric['energy']
        if metric.get('deduction') is not None:
            summary[key]['avg_deduction'] += metric['deduction']
    
    # Calculate averages
    for key in summary:
        count = summary[key]['count']
        if count > 0:
            summary[key]['avg_reward'] /= count
            summary[key]['avg_latency'] /= count if summary[key]['avg_latency'] > 0 else 1
            summary[key]['avg_energy'] /= count if summary[key]['avg_energy'] > 0 else 1
            summary[key]['avg_deduction'] /= count if summary[key]['avg_deduction'] > 0 else 1
    
    output_path = Path(output_csv)
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['agent', 'count', 'avg_reward', 'avg_latency', 'avg_energy', 'avg_deduction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for agent, stats in sorted(summary.items()):
            row = {'agent': agent}
            row.update(stats)
            writer.writerow(row)
    
    print(f"✓ Summary statistics saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    # Check if log_path provided as command line argument
    log_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Parse log and create visualizable CSV
    metrics, csv_path = parse_log_file(log_file_path=log_path_arg)
    
    if metrics:
        create_summary_stats(metrics)
        print(f"\nYou can now open these CSV files in Excel, pandas, or create plots!")
