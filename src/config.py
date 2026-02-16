import os

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing {name}. In Codespaces: Repo Settings → Secrets and variables → Codespaces."
        )
    return val
