"""zbuffer 核心 API（初始未实现）。"""


class ZBuffer:
    def __init__(self, w, h, near, far):
        """构造"""
        raise NotImplementedError

    def test(self, x, y, z, attr):
        """深度测试"""
        raise NotImplementedError

    def depth(self, x, y):
        """读深度"""
        raise NotImplementedError

    def attr(self, x, y):
        """读属性"""
        raise NotImplementedError
