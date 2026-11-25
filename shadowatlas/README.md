# shadowatlas

## 题面（可直接粘贴）

子目录：shadowatlas。阴影图集分配。约束：每帧分配不重叠且不越界；同尺寸请求合并到同一页（页面数最小化）；释放后可复用且不碎片化累积；按需扩容且旧页内容不丢；10 万次分配释放不超过 400 ms；非法尺寸与超额请求可区分报错。验收：cd shadowatlas && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd shadowatlas && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
