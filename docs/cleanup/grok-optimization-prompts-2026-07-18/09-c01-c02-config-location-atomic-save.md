# O10/C01-C02: 配置位置迁移与原子保存

## 交给 Grok 的提示词

基线 `f4c2a5e`。将配置目录与写入可靠性视为一个小的持久化事务任务。先审查 `config_manager.py`、启动调用点、现有 config tests；不要混入 UI/Provider/缓存重构，也不要接触用户真实配置。

目标：应用默认使用稳定的 app-data 配置位置；在首次发现旧 CWD 配置时进行一次、可恢复的迁移。所有保存都先写同目录临时文件，flush/close 后使用 `os.replace` 原子替换，写入失败时旧 config 完整保留。

迁移必须不覆盖已有新位置 config；失败要返回可操作的错误且不删除旧文件。改动前创建 Git 回退和文件备份，只用临时目录测试，独立提交。

## 建议实现路径

1. 集中定义 config path resolver，区分新 app-data、旧 CWD 与测试 override。
2. 添加幂等迁移函数：仅新路径不存在且旧路径有效时复制/replace。
3. 保存使用 `NamedTemporaryFile` 或同目录临时命名，显式 fsync（平台允许时）后 `os.replace`。
4. 清理仅本模块自己的 stale temporary family，绝不通配删除用户目录。

## 测试方案与验收

- 首次迁移成功；第二次不重复迁移；新位置已有 config 时旧文件不覆盖它。
- mock 写入或 replace 失败：旧 config 字节内容不变，临时文件得到受限清理。
- 路径 resolver 在测试环境使用隔离目录。
- 运行 config/startup 测试、完整 discover、`py_compile`、`git diff --check`。
