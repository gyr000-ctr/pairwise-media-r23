"""shadowatlas 核心实现：按尺寸类分组的页式阴影图集分配器。

设计要点：
- 每个页只服务一种尺寸类 (w, h)，页内按固定网格划分槽位，
  因此释放后复用不会产生碎片累积。
- 同尺寸请求优先填入已有空闲槽位的页（页面数最小化）；
  没有可用页时优先回收完全空闲的页，最后才按需扩容追加新页。
- 页一旦创建，其在页表中的下标永不改变，扩容只是追加，
  因此旧页内容（已分配矩形）不会丢失或移动。
- 所有可预期错误都继承 ValueError，同时按类型区分：
  非法尺寸 -> InvalidSizeError；超额请求 -> TooLargeError / AtlasFullError；
  非法释放 -> InvalidRectError。
- alloc/free/pages 均由同一把锁保护，并发安全。
"""

from __future__ import annotations

import threading

__all__ = [
    "ShadowAtlas",
    "AtlasError",
    "InvalidSizeError",
    "TooLargeError",
    "AtlasFullError",
    "InvalidRectError",
]


class AtlasError(ValueError):
    """shadowatlas 所有可预期错误的基类（同时是 ValueError）。"""


class InvalidSizeError(AtlasError):
    """非法尺寸：非整数、布尔、非有限值（inf/nan）或 <= 0。"""


class TooLargeError(AtlasError):
    """超额请求：单个请求超过页容量，放不进任何一页。"""


class AtlasFullError(TooLargeError):
    """达到 max_pages 上限，无法继续扩容。"""


class InvalidRectError(AtlasError):
    """释放格式非法、未知或已释放的矩形。"""


def _check_dim(value, name):
    """校验一个尺寸参数为正整数（拒绝 bool / float / nan / inf 等）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSizeError(
            f"{name} 必须是正整数, got {type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise InvalidSizeError(f"{name} 必须为正整数, got {value}")
    return value


class _Page:
    """单页：只服务一种尺寸类，槽位为固定网格。"""

    __slots__ = ("w", "h", "cols", "capacity", "next_slot", "freed", "used")

    def __init__(self, w, h, atlas_size):
        self.w = w
        self.h = h
        self.cols = atlas_size // w
        rows = atlas_size // h
        self.capacity = self.cols * rows
        self.next_slot = 0      # 从未使用过的下一个槽位
        self.freed = []         # 已释放、可复用的槽位（栈）
        self.used = 0           # 当前已分配槽位数


class ShadowAtlas:
    """阴影图集分配器。

    rect 格式为 4 元组 ``(page, x, y, w)``：页下标、页内偏移与请求宽度，
    对应的高度由分配器内部记录，free 时按整个元组精确匹配。
    """

    def __init__(self, size, max_pages=None):
        _check_dim(size, "size")
        if max_pages is not None:
            _check_dim(max_pages, "max_pages")
        self._size = size
        self._max_pages = max_pages
        self._pages = []        # list[_Page | None]，None 表示可回收的空页
        self._open = {}         # (w, h) -> 有空闲槽位的页下标列表（栈式使用）
        self._empty = []        # 完全空闲、可改作任意尺寸类的页下标（栈）
        self._allocs = {}       # rect -> (page_index, slot)
        self._lock = threading.Lock()

    def alloc(self, w, h):
        """分配一个 w x h 的矩形，返回 (page, x, y, w) 四元组。"""
        _check_dim(w, "w")
        _check_dim(h, "h")
        size = self._size
        if w > size or h > size:
            raise TooLargeError(
                f"请求 {w}x{h} 超过页容量 {size}x{size}，无法放入任何一页"
            )
        key = (w, h)
        with self._lock:
            open_list = self._open.get(key)
            if open_list:
                pi = open_list[-1]
                page = self._pages[pi]
            elif self._empty:
                # 回收一个完全空闲的页，改作当前尺寸类
                pi = self._empty.pop()
                page = _Page(w, h, size)
                self._pages[pi] = page
                open_list = self._open.setdefault(key, [])
                open_list.append(pi)
            else:
                # 按需扩容：只在尾部追加，旧页内容不受影响
                if self._max_pages is not None and len(self._pages) >= self._max_pages:
                    raise AtlasFullError(
                        f"页数已达上限 {self._max_pages}，无法为 {w}x{h} 扩容"
                    )
                pi = len(self._pages)
                page = _Page(w, h, size)
                self._pages.append(page)
                open_list = self._open.setdefault(key, [])
                open_list.append(pi)

            if page.freed:
                slot = page.freed.pop()
            else:
                slot = page.next_slot
                page.next_slot += 1
            page.used += 1
            if page.used == page.capacity:
                open_list.pop()  # 页满，移出开放列表（它一定是栈顶）

            x = (slot % page.cols) * w
            y = (slot // page.cols) * h
            rect = (pi, x, y, w)
            self._allocs[rect] = (pi, slot)
            return rect

    def free(self, rect):
        """释放此前 alloc 返回的矩形；未知/重复/非法矩形抛 InvalidRectError。"""
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            raise InvalidRectError(f"非法矩形: {rect!r}")
        key = tuple(rect)
        for v in key:
            if isinstance(v, bool) or not isinstance(v, int):
                raise InvalidRectError(f"非法矩形分量: {rect!r}")
        with self._lock:
            rec = self._allocs.pop(key, None)
            if rec is None:
                raise InvalidRectError(f"未知或已释放的矩形: {rect!r}")
            pi, slot = rec
            page = self._pages[pi]
            page.used -= 1
            page.freed.append(slot)
            if page.used == 0:
                # 整页空闲：从开放列表摘除，放入回收池（页下标保留，内容已空）
                ol = self._open.get((page.w, page.h))
                if ol:
                    if ol[-1] == pi:
                        ol.pop()
                    else:
                        try:
                            ol.remove(pi)
                        except ValueError:
                            pass
                self._pages[pi] = None
                self._empty.append(pi)
            elif page.used == page.capacity - 1:
                # 从满变非满，重新加入开放列表
                self._open.setdefault((page.w, page.h), []).append(pi)

    def pages(self):
        """当前页数（只增不减，空页保留在回收池中复用）。"""
        with self._lock:
            return len(self._pages)
