# O8: Capture worker 只读取不可变 UI 快照

## 交给 Grok 的提示词

基线 `f4c2a5e`。实现一次单独的线程边界修复：capture worker 在热路径不能直接读取 Tk `Variable`、widget、`winfo_exists()` 或 overlay geometry。先审查 `worker_capture.py`、`app_capture_ocr.py`、overlay move/resize 回调与现有 worker 测试；不要把 O7 Preview 改动混入。

目标：UI 线程发布不可变 capture snapshot，包含 geometry、OCR model、scan interval、必要的 OCR settings 和 generation。worker 在一轮循环开始时只读取一个 snapshot；generation 变化使旧 OCR state/帧失效。

约束：snapshot 不得保存 widget 引用或 `tk.Variable`；采用短锁或原子替换；不改变“丢旧留新”的实时队列策略；不读取或输出用户文字、key、完整配置。改动前先建立 Git 回退与文件备份，TDD 后单独提交。

## 建议实现路径

1. 定义小型 frozen dataclass/tuple，并确定唯一 UI 线程发布点。
2. 在 overlay move/resize、设置保存和 OCR mode 变更处更新 generation + snapshot。
3. worker 通过 app 的纯 Python getter 获取 snapshot，不访问 Tk 对象。
4. generation 变化时仅清理与旧 geometry/settings 绑定的 OCR frame state。

## 测试方案与验收

- 使用不含 Tk widget 的 fake app，worker 仍可完成一轮 capture 路由。
- geometry/model 变更使旧 generation 的帧失效。
- shutdown/overlay destroy 时没有 TclError。
- 静态或 mock 断言 capture 热路径不调用 Tk Var/widget API。
- 运行 `tests.test_paddle_ocr_backend`、worker/capture 相关测试、完整 discover、`py_compile`、`git diff --check`。
