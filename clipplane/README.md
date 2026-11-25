# clipplane

## 题面（可直接粘贴）

子目录：clipplane。多边形裁剪平面。约束：平面两侧分类与点积一致（≤1e-12）；裁剪后多边形在平面内侧；裁剪不改变绕序；顶点数守恒校验（进出点成对）；退化多边形返回空；1 万次不超过 300 ms；非法平面可区分报错。验收：cd clipplane && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd clipplane && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
