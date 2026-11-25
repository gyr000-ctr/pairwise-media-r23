"""shadowatlas 核心 API（初始未实现）。"""


class ShadowAtlas:
    def __init__(self, size):
        """构造"""
        raise NotImplementedError

    def alloc(self, w, h):
        """分配"""
        raise NotImplementedError

    def free(self, rect):
        """释放"""
        raise NotImplementedError

    def pages(self):
        """页数"""
        raise NotImplementedError
