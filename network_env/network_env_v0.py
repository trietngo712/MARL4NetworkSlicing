from .env.network_env import NetworkEnv

def env(**kwargs):
    return None

def parallel_env(config_path = None, traffic_path = None, log_path = None, num_agents = None, **kwargs):
    return NetworkEnv(config_path=config_path, num_agents=num_agents, traffic_path=traffic_path, log_path=log_path, **kwargs)