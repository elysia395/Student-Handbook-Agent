# Student-Handbook-Agent

广州中医药大学校园智能助手 —— 基于 RAG 的学生手册问答系统。

## 项目初衷

本校的学生手册有足足两百多页，内容繁杂，光是翻开目录就让人望而却步。为了让同学们能够快速查阅规章制度、奖学金评定、请假流程等常用信息，我将整本学生手册构建成 RAG 知识库，做了一个简单的问答 Agent。同时也是为了检验自己对 RAG 技术的学习成果。

## 功能介绍

- 本地知识库问答：基于《广州中医药大学学生手册（2024年版）》进行精准检索和回答
- 联网搜索：实时搜索校园新闻、通知、活动等信息
- 多轮对话：支持上下文记忆，最多保留 10 轮对话
- 来源标注：每条回答自动标注信息来源（学生手册 / 网络搜索）
- 灵活配置：LLM 不限于特定厂商，支持任何兼容 OpenAI API 的模型

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | 支持 OpenAI API 的任意模型（默认 DeepSeek V4 Flash） |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量数据库 | ChromaDB（余弦距离） |
| 联网搜索 | Tavily API |
| PDF 解析 | PyMuPDF + Tesseract OCR |

## 快速开始

### 1. 环境要求

- Python >= 3.12
- uv（推荐，用于管理虚拟环境和依赖）
- Tesseract OCR（用于 PDF 文字识别）
  - Windows: 下载安装 Tesseract-OCR，安装路径默认为 `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

### 2. 克隆项目

```bash
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 配置 API Key

本项目需要 LLM API Key 和 Tavily API Key，请自行申请。

#### 第一步：申请 LLM API Key

支持任何兼容 OpenAI API 的模型服务，例如：

| 厂商 | 注册地址 | 模型示例 |
|------|----------|----------|
| DeepSeek | https://platform.deepseek.com/ | deepseek-v4-flash、deepseek-chat |
| 硅基流动 SiliconFlow | https://siliconflow.cn/ | Qwen、DeepSeek 等开源模型 |
| OpenAI | https://platform.openai.com/ | gpt-4o、gpt-3.5-turbo |
| 阿里云通义千问 | https://dashscope.aliyun.com/ | qwen-plus、qwen-turbo |

#### 第二步：申请 Tavily API Key（联网搜索）

1. 访问 https://tavily.com/
2. 注册账号，免费套餐每月 1000 次搜索
3. 在 Dashboard 中创建 API Key

#### 第三步：创建 .env 文件

在项目根目录创建 `.env` 文件，按你选择的厂商填写对应变量名和 Key。变量名可以自定义，只需与 `main.py` 中 `os.getenv("你的变量名")` 对应即可。

```bash
# 示例：使用 DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxxxxxx

# 示例：使用 硅基流动
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxxxxxx

# 示例：使用 OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxxxxxx
```

#### 第四步：配置 LLM 连接

打开 `main.py`，修改以下两处：

```python
llm_client = OpenAI(
    api_key=os.getenv("你的APIKey变量名"),  # 与 .env 中的变量名对应
    base_url="你的API地址"  # 例如 https://api.deepseek.com 或 https://api.siliconflow.cn/v1
)

# ...

model="你的模型名称"  # 例如 deepseek-v4-flash、Qwen/Qwen2.5-7B-Instruct
```

> **安全提醒**：`.env` 文件已加入 `.gitignore`，不会被提交到 git 仓库。请妥善保管你的 API Key，切勿泄露。

### 5. 导入知识库

将学生手册 PDF 放入 `knowledge/` 目录，然后运行：

```bash
uv run python -m RAG.ingest knowledge/学生手册（2024年版）.pdf
```

首次运行会进行 OCR 处理和向量化，约需 15-20 分钟。

### 6. 启动助手

```bash
uv run python main.py
```

启动后即可开始对话。

| 命令 | 说明 |
|------|------|
| 输入问题 | 开始对话 |
| `/reset` | 清空对话历史 |
| `exit` | 退出程序 |

## 使用示例

```
你：广州中医药大学的建校时间
🤖 助手：广州中医药大学肇始于1924年，1956年经国务院批准成立广州中医学院，1995年正式更名为广州中医药大学。(来源：学生手册)

你：处分种类有哪些
🤖 助手：学校对犯有错误的学生视情节轻重给予批评教育或纪律处分。处分种类分为：警告、严重警告、记过、留校察看、开除学籍。(来源：学生手册)

你：今天的校园新闻有什么
🤖 助手：... (来源：网络搜索)
```

## 对话历史说明

- 系统自动保留最近 10 轮对话上下文
- 输入 `/reset` 可随时清空历史
- 程序重启后历史不会保留（纯内存模式）

## 项目结构

```
make-agent/
├── main.py              # Agent 主入口（LLM Function Calling + 对话管理）
├── RAG/
│   ├── retriever.py     # 向量检索 + Rerank 管线
│   ├── Embedder.py      # BAAI/bge-m3 Embedding 模型
│   └── ingest.py        # PDF OCR + 知识库导入
├── knowledge/           # 存放学生手册 PDF
├── chroma_db/           # ChromaDB 向量数据库（自动生成）
├── models/              # 本地模型缓存（自动生成）
├── .env                 # API Key 配置文件（需自行创建）
├── .gitignore           # Git 忽略配置
├── pyproject.toml       # Python 依赖配置
└── uv.lock              # uv 锁定文件
```

## 检索流程

```
用户问题 → 向量粗召(top 20) → 余弦初筛(≥0.35)
  → Reranker 精排(≥0.45) → 动态截取
  → LLM 生成回答
```

## 常见问题

**Q: 运行报错 "No module named 'chromadb'"？**
A: 确保已运行 `uv sync` 安装所有依赖。

**Q: OCR 识别效果不好？**
A: 确保 Tesseract OCR 已安装且路径正确。Windows 下默认路径为 `C:\Program Files\Tesseract-OCR\tesseract.exe`。

**Q: 回答不准确？**
A: 尝试用 `/reset` 清空对话历史后重新提问，或检查 PDF 是否已正确导入知识库。

**Q: API Key 会泄露吗？**
A: `.env` 文件已在 `.gitignore` 中排除，不会被提交到 git 仓库。请勿将 API Key 硬编码在代码中。

**Q: 如何切换 LLM 模型？**
A: 修改 `main.py` 中的 `base_url` 和 `model` 参数即可。任何兼容 OpenAI API 格式的服务都可以使用。

