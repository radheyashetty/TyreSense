import yaml


def load_config(config_path="config.yaml"):
    """Read project configuration from yaml file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
