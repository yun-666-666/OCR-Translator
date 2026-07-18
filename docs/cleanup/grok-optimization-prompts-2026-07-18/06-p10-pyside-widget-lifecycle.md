# O10/P10: PySide overlay 的 exists 生命周期契约

## 交给 Grok 的提示词

基线 `f4c2a5e`。只修复 PySide overlay 的 `winfo_exists()` 兼容契约。先阅读 `pyside_overlay.py` 的两个 wrapper、调用点和 overlay startup tests。不要全局替换 Tk 的 `winfo_exists()`，不要把函数写成恒真，也不要改隐藏窗口的语义。

目标：destroy 后 `winfo_exists()` 必须返回 false；hide 后 overlay 仍可用并返回 true；Qt `destroyed`/对象有效性与 wrapper 自己的生命周期状态必须共同决定结果。

改动前创建 Git 回退与文件备份；新增确定性测试后只做这个小改动并独立提交。

## 建议实现路径

1. 为每个 wrapper 维护 destroyed 标记，并连接 Qt destroyed signal。
2. 若可用，使用 Qt 对象有效性检查作为第二道保护。
3. hide/show 只影响 viewable，不影响 exists。
4. 所有异常路径安全返回 false，不访问已经销毁的 Qt 对象。

## 测试方案与验收

- 两种 wrapper：新建 true、hide 后 true、destroy 后 false。
- destroy 后调用不抛 Qt runtime error。
- 现有 Tk overlay 生命周期不回归。
- 运行 overlay/startup 测试、完整 discover、`py_compile`、`git diff --check`。
