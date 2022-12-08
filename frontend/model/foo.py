import os

# set some global variable constants
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "sk_model")

def print_env_vars():
    """Echo back what the env vars are set to"""
    env_vars = [
        MLFLOW_REGISTRY_URI,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXP_NAME,
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_ARTIFACT_PATH,
    ]
    env_var_names = [
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXP_NAME",
        "MLFLOW_REGISTERED_MODEL",
        "MLFLOW_ARTIFACT_PATH",
    ]
    for name, env_var in zip(env_var_names, env_vars):
        print(f"{name}:\t{env_var}")

def print_inside_envs():
    """Get env vars inside the func, then print"""
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXP_NAME = os.getenv("MLFLOW_EXP_NAME", "nba-leaguedash-cluster")
    MLFLOW_REGISTERED_MODEL = os.getenv("MLFLOW_REGISTERED_MODEL", "nba-player-clusterer")
    MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "sk_model")

    env_vars = [
        MLFLOW_REGISTRY_URI,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXP_NAME,
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_ARTIFACT_PATH,
    ]
    env_var_names = [
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXP_NAME",
        "MLFLOW_REGISTERED_MODEL",
        "MLFLOW_ARTIFACT_PATH",
    ]
    for name, env_var in zip(env_var_names, env_vars):
        print(f"{name}:\t{env_var}")