# O10/P12-P15: legacy provider 调用惯性化与历史文档归档

## 交给 Grok 的提示词

基线 `f4c2a5e`。只做 live-path 与历史文档一致性收尾。先检查 `handlers/translation_results.py` 的 legacy adapter、启动入口、`tests/test_live_path_inventory.py`、`tests/test_custom_ai_startup.py` 与受影响文档。不要删除代码或重写 provider 架构，除非测试证明调用仍可达。

目标：遗留 provider 调用在冷启动和正常 live path 中保持 inert（不可达且无副作用）；历史文档明确标记为 archived/legacy，避免让用户以为仍受支持。不要把未验证的 provider 依赖重新加回打包配置。

改动前创建 Git 回退与文件备份，先增加 live-path inventory 测试，再做最小改动；只提交验证过的源、文档、测试和 handoff。

## 建议实现路径

1. 建立当前产品 live path 清单：PaddleOCR/Custom AI OCR、缓存、Custom AI 翻译、overlay。
2. 对 legacy entrypoints 采用无副作用 return/明确 archived adapter，保留兼容调用但不初始化旧 client。
3. 给历史文档添加简短 archived 标记与指向当前路径的链接；不大规模重写翻译文档。
4. 让测试检查 source inventory、startup import 与打包 spec 都不重新依赖 legacy provider。

## 测试方案与验收

- 冷启动不实例化 legacy client，也不读取 legacy key。
- live-path inventory 和 PyInstaller/spec 测试一致。
- 文档标记不影响当前用户手册入口。
- 运行 startup/inventory 测试、完整 discover、`py_compile`、`git diff --check`。
