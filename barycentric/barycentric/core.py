"""barycentric 核心 API（初始未实现）。"""


class Barycentric:
    def coords(self, tri, p):
        """重心坐标"""
        raise NotImplementedError

    def inside(self, tri, p):
        """是否在内"""
        raise NotImplementedError

    def perspective(self, tri, w, p):
        """透视校正权重"""
        raise NotImplementedError
