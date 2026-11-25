from .core import *  # noqa: F401,F403
from . import core as _core


def __getattr__(name):
    # 未在 core.__all__ 中显式列出的异常别名等，交给 core 的动态工厂
    return _core.__getattr__(name)
