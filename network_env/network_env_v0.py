from .env.network_env import NetworkEnv

def env(**kwargs):
    return None

def parallel_env(config_path = None, **kwargs):
    return NetworkEnv(config_path=config_path, **kwargs)