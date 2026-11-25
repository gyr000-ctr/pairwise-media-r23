# barycentric

## 题面（可直接粘贴）

子目录：barycentric。重心坐标与光栅化覆盖。约束：重心坐标和为 1（≤1e-12）；三角形内外判定与边函数符号一致；退化三角形报错；顶点处重心为 (1,0,0) 置换；透视校正插值权重与深度倒数一致（≤1e-9）；1 万三角形覆盖测试不超过 300 ms；并发安全。验收：cd barycentric && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd barycentric && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
