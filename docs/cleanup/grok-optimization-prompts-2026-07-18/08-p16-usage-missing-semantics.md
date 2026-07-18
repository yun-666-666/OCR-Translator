# O10/P16: 缺失 usage 不得写成零 token

## 交给 Grok 的提示词

基线 `f4c2a5e`。只实现 usage 缺失语义修复。先审查 `custom_ai_requests.py` 的 usage 归一化、`handlers/translation_results.py` 的 short-log 记录与已有 Custom AI 测试。不要改变真实 usage 的成本计算、缓存 key 或 provider wire payload。

目标：provider stream/response 没有 usage 时，在短日志和可观测数据中明确标为 `n/a` 或 `usage_missing`，不能把未知值伪装为 `0 token` 或 `$0`。有真实 usage 时保持现在的数值格式与行为。

改动前创建 Git 回退与文件备份；先写无 usage 的失败测试，最后单独提交。

## 建议实现路径

1. 在 usage 归一化层保留“字段缺失”和“数值零”之间的区别。
2. short-log formatter 只在数值已知时格式化为数；未知输出固定、机器可检索的标记。
3. runtime metrics 也区分 unknown，避免把 unknown 纳入零成本均值。
4. 保持 O1：默认短日志仍绝不包含 OCR/翻译正文。

## 测试方案与验收

- stream 无 usage：日志含 `usage_missing`/`n/a`，不含 `Input Tokens: 0`、`Output Tokens: 0` 或伪造 `$0`。
- 明确零 token（如有合法场景）仍按 0 表示。
- 完整 usage 保持既有 token/cost 输出。
- 运行 Custom AI/logging 测试、完整 discover、`py_compile`、`git diff --check`。
