import os
from typing import Optional

from dotenv import load_dotenv, find_dotenv


def load_environment(env_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file.

    If env_path is None, it will search upwards from CWD to find the first .env.
    """
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(dotenv_path=found, override=False)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get env var after .env loading. If required and missing, raise error."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"缺少必要的环境变量: {name}")
    return value


__all__ = ["load_environment", "get_env"]
