# O6: PaddleOCR prewarm 冷启动观测（先不改策略）

## 交给 Grok 的提示词

基线 `f4c2a5e`。这是纯观测增量，不允许以单次慢启动样本更改模型、关闭预热、引入长期 daemon 或修改 20 秒默认等待。先审查 `app_logic.py` 的 prewarm state、`worker_capture.py` 等待路径、`paddle_ocr_backend.py` 与 `tests/test_paddle_ocr_backend.py`。

目标：补充不含用户文本的结构化 prewarm 观测，以判断 text-recognition 初始化慢时究竟是否阻塞 Start 或首次 OCR。至少记录 generation、reason、settings 的安全摘要、启动方式、是否等待 ready、各 phase 耗时、首次 OCR 等待、设备/CPU 信息；明确区分模型下载、文件访问、模型构建等可判断阶段。

不得记录 OCR 文本、路径中的敏感用户信息、API key 或完整环境变量。只做指标和测试，不做基于指标的策略改动。改动前创建 Git 回退与文件备份，并提供一份冷启动/热启动采样说明。

## 建议实现路径

1. 在现有 prewarm generation/state 上增加有限、线程安全的 metrics snapshot。
2. 在 schedule/start/wait/completed/first-use 边界记录 monotonic duration 与 reason code。
3. 把 metrics 暴露给现有 diagnostics 摘要或受限 debug 日志，不增加高频日志。
4. 在测试中用 mock monotonic 与 fake engine 区分 phase，不依赖真实 57 秒初始化。

## 测试方案与验收

- 冷、热、已 ready、settings 变更、失败 prewarm 都生成正确的有限指标。
- Start 路径与 worker 等待路径可从指标区分，不误报为同一耗时。
- 无 OCR 文本进入日志/metrics。
- 运行 `tests.test_paddle_ocr_backend`、startup/overlay 相关测试、完整 discover、`py_compile`、`git diff --check`。
- 交付两次真实采样（冷/热）或明确说明为何当前环境无法产生真实数据。
