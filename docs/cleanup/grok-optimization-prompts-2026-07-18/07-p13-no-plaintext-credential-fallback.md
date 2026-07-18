# O10/P13: credential store 失败时禁止明文 key fallback

## 交给 Grok 的提示词

基线 `f4c2a5e`。这是安全修复，只检查 `credential_store.py`、`custom_ai_profiles.py` 及相关测试。绝不打印、提交或在报告中复述本机 profile/API key；不要修改 provider 请求或 profile UI。

目标：credential store 写入/更新失败时，profile JSON 不得回退写入明文 key。保留旧的已保存 profile 状态，并向调用方返回可操作、脱敏的错误；内存和磁盘状态都不能出现半提交。

改动前创建 Git 回退与文件备份，先写失败路径测试。把敏感信息扫描加入验证，且只提交源代码、测试和 handoff。

## 建议实现路径

1. 审核 add/update 的 credential write 与 profile JSON save 顺序。
2. 将 credential 写入失败建模为原子事务失败：不发布内存 state、不替换 JSON。
3. 保持已有 credential ref 指向，不生成 `api_key` 字段或明文 fallback。
4. 所有错误信息只给出 provider/profile 标识的安全摘要。

## 测试方案与验收

- mock credential store add/update 失败后，JSON 无新增/替换明文 key。
- 已存在 profile 在失败后内容与 credential ref 保持原样。
- 内存快照未发布半更新状态。
- staged diff 对 `api_key`/常见 key 模式做脱敏扫描；完整 discover、`py_compile`、`git diff --check`。
