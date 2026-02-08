import os


def has_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def is_github_actions() -> bool:
    """Check if running in GitHub Actions CI."""
    return os.getenv("GITHUB_ACTIONS") == "true"


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_hunyuan3d_installed() -> bool:
    """Check if Hunyuan3D-2 is installed for 3D geometry generation."""
    try:
        from hy3dgen.shapegen.pipelines import export_to_trimesh  # noqa: F401

        return True
    except ImportError:
        return False
