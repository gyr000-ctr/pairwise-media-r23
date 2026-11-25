"""clipplane 核心 API（初始未实现）。"""


class ClipPlane:
    def classify(self, p, plane):
        """分类"""
        raise NotImplementedError

    def clip(self, poly, plane):
        """裁剪"""
        raise NotImplementedError
