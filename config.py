import tomllib  
import argparse
from pathlib import Path

def loadConfig(config_path=None):
    """Carrega configuração do arquivo TOML"""
    if config_path is None:
        config_path = "config.toml"
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config