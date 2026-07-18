# O7: Preview 不在 Tk 主线程执行完整 OCR

## 交给 Grok 的提示词

基线 `f4c2a5e`。只实现 Preview OCR 的 UI 响应优化；不要与 capture snapshot、预热、provider、主 OCR worker 或 UI 重构混在同一个提交。先检查 `app_capture_ocr.py` 的 `preview_realtime_update`、Preview 生命周期与相关测试。

目标：Tk 的 `after` 回调不得同步执行完整 PaddleOCR。默认 Preview 显示最新图像和主 OCR worker 的最近结果；若保留独立 Preview OCR，最多一个在途任务，新帧只覆盖 pending frame，worker 完成后仅用 `root.after` 回到 UI。

约束：关闭 Preview 时递增 generation，使旧 future 完成回调失效；worker 不得直接操作任何 Tk widget；不能每 500ms 无限制提交 future；不得改变主 OCR 提交/缓存/翻译行为。改动前创建 Git 回退分支和文件备份，结束时独立提交。

## 建议实现路径

1. 把“抓取/显示图片”和“独立 OCR”分离；after 回调仅做轻量 UI 与任务调度。
2. 使用单 worker executor 或等价的单在途状态；保留一个可替换的最新 pending frame。
3. 给窗口关闭、停止和设置变更加入 generation token。
4. 回调进入 UI 前再次确认窗口仍存在、generation 相同。

## 测试方案与验收

- mock OCR，断言 after callback 不同步调用 OCR。
- 高频刷新时最多一个 in-flight 和一个最新 pending frame，旧帧不会排队累积。
- 关闭 Preview 后完成的旧 future 不写 UI，也不抛 TclError。
- 手动 smoke：打开 Preview、拖主窗口、Stop、关闭窗口时 UI 无明显卡顿。
- 运行 Preview/overlay 相关测试、`tests.test_paddle_ocr_backend`、完整 discover、`py_compile` 与 `git diff --check`。
