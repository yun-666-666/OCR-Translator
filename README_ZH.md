# Game-Changing Translator

**中文** | [English](README.md)

> **原项目与原作者：** [Game-Changing Translator](https://github.com/tomkam1702/OCR-Translator)，作者 [Tomasz Kamiński](https://github.com/tomkam1702)。本仓库为修改后的 Fork，完整保留原作者版权与署名。

Game-Changing Translator 是一款 Windows 桌面 OCR 实时翻译工具，适用于无法直接复制的屏幕文字。选择屏幕中的文字区域与译文叠加区域后，应用会持续完成截图、识别、翻译、缓存并显示结果。

## 当前权威基线

- **RapidOCR 本地 OCR**：默认 OCR，也是当前主项目的权威本地识别路径。
- **PaddleOCR 本地 OCR**：可选高级引擎，提供更完整的模型与识别参数。
- **Custom AI 配置档**：支持 OpenAI 兼容的翻译端点，并可选用 Custom AI OCR 处理图像型提供商。

后续仅用于个人机器加速的本地翻译模型实验不属于此权威基线。本项目通过 Custom AI 配置档接入本地或远程 OpenAI 兼容翻译服务。

## 主要功能

- 自选屏幕区域作为 OCR 输入和译文叠加输出
- 实时 OCR 与翻译循环，并防止过期结果覆盖新结果
- 默认 RapidOCR 后端，可配置最低置信度
- 可选 PaddleOCR 后端，可配置语言、OCR 版本、模型大小、设备、置信度、放大倍率、检测限制与文本行方向
- 支持 WebP、PNG、JPEG 图像载荷的 Custom AI OCR
- 通过 OpenAI 兼容配置档进行 Custom AI 翻译
- 统一翻译缓存，减少重复 API 请求
- 可自定义翻译提示词、字体、颜色、透明度和叠加窗口位置
- 全局快捷键控制开始、暂停和恢复

## 安装与运行

### 环境要求

- Windows 10 或 Windows 11
- 从源码运行时使用 Python 3.9-3.12

发布版不需要另行安装 OCR 引擎。源码安装默认使用 RapidOCR；`requirements.txt` 同时包含受支持的 PaddleOCR CPU 依赖。

### 从源码运行

```bash
git clone https://github.com/yun-666-666/OCR-Translator.git
cd OCR-Translator
python -m pip install -r requirements.txt
python main.py
```

## 快速开始

1. 启动应用。
2. 选择 OCR 文字区域。
3. 选择译文叠加显示区域。
4. 默认保留 RapidOCR；需要高级控制时再选择 PaddleOCR。
5. 配置用于翻译的 OpenAI 兼容 Custom AI 配置档。本地无认证端点可以不填写 API Key。
6. 选择端点协议：**Chat Completions** 或 **Responses API**。
7. 点击 **Start** 开始翻译。

PaddleOCR 只会在实际选择并首次使用时加载，不再随应用启动自动预热。需要 CUDA 加速时，请按 PaddlePaddle 官方安装矩阵，用与本机 CUDA 匹配的 `paddlepaddle-gpu` 替换 CPU 版 `paddlepaddle`。

## 配置与隐私

请勿提交真实 API Key、个人服务商配置、调试日志、运行缓存或自动生成的本地配置。`ocr_translator_config.ini`、日志、缓存与本地备份目录均应保持在 Git 之外。

请使用 `ocr_translator_config.example.ini` 作为可公开的配置模板。相对路径的 Custom AI 配置档、`custom_prompt.txt` 与翻译缓存均固定相对于应用/项目根目录，避免从不同工作目录启动时读到不同文件。

## 开发与测试

在开发环境中额外安装 `pytest` 后，从项目根目录运行：

```bash
python -m pytest
```

`pytest` 只是开发测试工具，不影响应用启动或翻译运行。

## 文档

- [English documentation](README.md)
- [用户手册](docs/user-manual.html)
- [安装指南](docs/installation.html)
- [故障排除](docs/troubleshooting.md)
- [开发者指南](docs/developer-guide.md)
- [更新日志](CHANGELOG.md)

## 许可证

本修改版 Fork 使用 GNU General Public License v3 或更高版本（GPL-3.0-or-later），并保留原项目的版权声明与 GPL 许可证。完整条款和第三方声明请见 [LICENSE](LICENSE)。

若你发布修改版或二进制文件，请遵守 GPL，并保留 [ATTRIBUTION.md](ATTRIBUTION.md) 中的原项目与原作者署名。

## 致谢

- **GPT**：本 Fork 开发工作的主要 AI 协作者。
- **Claude** 与 **Grok**：参与实现、审查和迭代的 AI 协作者。
- **Tomasz Kamiński**：Game-Changing Translator 原作者与维护者；本项目基于其原始工作继续开发。

完整署名记录请见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

## 贡献

欢迎提交贡献。提交 Pull Request 前，请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并保留所有版权、许可证和原作者署名。

本 Fork 源自 [Tomasz Kamiński 的 Game-Changing Translator](https://github.com/tomkam1702/OCR-Translator)。派生项目请保留这一来源说明与原作者署名。
