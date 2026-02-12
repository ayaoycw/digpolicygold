# Browse Use 测试项目

按 [browser-use 官方 GitHub](https://github.com/browser-use/browser-use) 说明准备的环境。

## 环境准备（已完成）

1. **uv 初始化**：`uv init`
2. **安装 browser-use**：已加入 `pyproject.toml`，执行 `uv sync` 完成安装
3. **Chromium**：已通过 `uv run browser-use install` 安装
4. **API Key**：需在 [Browser Use Cloud](https://cloud.browser-use.com/new-api-key) 获取并配置

## 使用前配置

1. 复制环境变量示例并填入 API Key：
   ```powershell
   copy .env.example .env
   ```
2. 编辑 `.env`，将 `your-key` 替换为你的 `BROWSER_USE_API_KEY`。

## 运行测试

运行官方示例（会打开浏览器并查询 browser-use 仓库的 star 数）：

```powershell
uv run python example_agent.py
```

若终端出现编码错误，可先设置 UTF-8 再运行：

```powershell
$env:PYTHONIOENCODING="utf-8"; uv run python example_agent.py
```

## 快速生成更多模板

```powershell
uvx browser-use init --template default
uvx browser-use init --template advanced
uvx browser-use init --template tools
```

## Web UI（官方界面）

项目内已克隆 **browser-use 官方 Web UI**，用于在浏览器里通过 Gradio 界面跑 AI Agent。

- 位置：**`web-ui/`** 目录（来自 [browser-use/web-ui](https://github.com/browser-use/web-ui)）
- 环境：已创建 `.venv`、安装依赖、安装 Playwright Chromium，并已复制 `.env.example` 为 `.env`
- 配置：编辑 **`web-ui/.env`**，填入至少一个 LLM 的 API Key（如 `OPENAI_API_KEY`、`DEFAULT_LLM=openai`）
- 启动与测试：详见 **`web-ui/启动WebUI.md`**

快速启动：

```powershell
cd web-ui
.\.venv\Scripts\python.exe webui.py --ip 127.0.0.1 --port 7788
```

浏览器访问 **http://127.0.0.1:7788** 即可使用。

## 参考

- 仓库：https://github.com/browser-use/browser-use
- Web UI 仓库：https://github.com/browser-use/web-ui
- 文档：https://docs.browser-use.com
- Cloud：https://cloud.browser-use.com
