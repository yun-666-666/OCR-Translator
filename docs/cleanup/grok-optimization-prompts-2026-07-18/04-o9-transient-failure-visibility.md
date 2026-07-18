# O9: 瞬时 provider 故障的节流可见性

## 交给 Grok 的提示词

基线 `f4c2a5e`。只增加“连续瞬时失败时的非侵入状态提示”，不要改 failover、cooldown、timeout、字幕显示内容或每帧日志。先查 `worker_translation.py`、状态栏更新路径、现有 transient-error 抑制逻辑和测试。

目标：首次/偶发故障仍保持旧字幕且不提示；同一 provider/session 连续 N 次失败或持续 T 秒后，在状态栏显示简短、节流、不可泄露的状态；首次成功立即清除；Stop/Close 后不再更新 UI。

实现必须把状态机限定在 provider/session，并用 request sequence/generation 防止旧请求覆盖新状态。不要显示 endpoint、请求正文、异常原文或凭据。改动前创建 Git 回退与文件备份，先测试后实现，最后独立提交。

## 建议实现路径

1. 使用小型 failure visibility state：streak、first failure time、last shown time、sequence/provider key。
2. 只在 UI 线程投递状态栏修改，回调前验证 app 仍运行且 sequence 仍最新。
3. 成功、Stop、provider 切换时重置对应状态。
4. 复用现有节流/日志设施，避免每帧刷新。

## 测试方案与验收

- 单次失败不产生状态提示。
- 达到次数或时间阈值后只提示一次/按节流频率提示。
- 首次成功清除提示；Stop 后旧回调不得更新 UI。
- 固定 5xx 或离线 mock 时用户可见状态，但字幕不闪烁且不刷屏。
- 运行 translation/显示相关测试、完整 discover、`py_compile`、`git diff --check`。
