# O3: 翻译 failover 共享 deadline

## 交给 Grok 的提示词

你正在审查并实现 OCR-Translator 的一个单独优化。基线是 `f4c2a5e`；O1（短日志隐私）、O2（显式备用 profile）、O4/O5（race health 与严格错误分类）均已完成，不要重做或扩展它们。

目标：让一次翻译请求的所有串行 failover 候选共享同一总 deadline，而不是每个候选各拿完整 timeout。先阅读 `handlers/translation_requests.py`、请求 snapshot 创建处、`worker_translation.py` 和现有 `tests/test_custom_ai.py`。先写会失败的确定性测试，再做最小实现。

实现要求：

- 在不可变 request snapshot 建立 `deadline_monotonic`。
- 每次即将发起候选前，检查 stopped、obsolete 与 deadline；候选 timeout 为 `max(0, deadline_monotonic - monotonic_now)`。
- 预算耗尽时不再发后续候选，返回统一、可诊断但不泄露凭据的 timeout 结果。
- 保持事实边界：不能取消已在运行的 HTTP 请求；只阻止未来 fallback 与陈旧显示。
- 添加只含计数/状态的可观测字段，例如已启动候选数、deadline 耗尽、obsolete abort；不要记录正文或 key。
- 不改 profile 资格、race、缓存 key、默认 10 秒策略或队列 overflow 语义。

每次改动前创建 Git 回退分支和文件备份；不要暂存本地 config、profile、日志或现有未跟踪文档。完成后提供精确 diff、测试结果和一个独立提交。

## 建议实现路径

1. 找到 request snapshot 的构造与 `_custom_ai_translate_with_failover` / 候选循环。
2. 将 deadline 随 snapshot 传递，避免在每个候选重新计算完整 timeout。
3. 把 preflight 检查收敛为一个小 helper，确保 safe/stream/fallback 路径采用相同边界。
4. 仅为新的计数/状态补运行指标或节流日志；不要写正文。

## 测试方案与验收

- 固定 monotonic 时钟：两个候选共享预算，第二个只能获得剩余时间。
- 第一候选耗尽全部预算：第二候选完全不发请求。
- 新 translation sequence 开始后：后续 fallback 不发。
- N 个失败 profile 的总等待不超过一次 request deadline 加小测试容差。
- 运行：`python -m unittest tests.test_custom_ai tests.test_latency_optimization -v`、完整 discover、改动文件 `py_compile`、`git diff --check`。
