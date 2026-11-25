"""msaa 核心 API（初始未实现）。"""


class Msaa:
    def __init__(self, samples):
        """构造"""
        raise NotImplementedError

    def coverage(self, tri, px, py):
        """覆盖率"""
        raise NotImplementedError

    def resolve(self, samples):
        """解析颜色"""
        raise NotImplementedError
