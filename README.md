# SciFlow - 基于DeepSeek的PDF文档RAG问答系统
> 课程设计 / 个人大作业项目，基于FastAPI + LangChain + Chroma向量库 + DeepSeek大模型，实现上传PDF文档，基于文档内容智能问答。
> 项目支持：PDF文档解析、文本分块、向量化存储、向量检索、大模型问答；可区分文档内知识，拒绝编造幻觉。

## 技术栈
后端：Python 3.10+, FastAPI, Uvicorn
LLM：DeepSeek API
RAG框架：LangChain
向量数据库：Chroma
嵌入模型：all-MiniLM-L6-v2 (Sentence-Transformers)
文档解析：PyPDFLoader

## 项目架构
1. 文件上传模块：接收前端上传PDF，保存到uploads文件夹
2. 文档处理模块：加载PDF、文本分割、生成向量
3. 向量存储模块：Chroma本地向量库，持久化存储向量，目录 vector_db
4. 检索增强模块：用户问题向量化，检索最相关文档片段
5. LLM问答模块：把检索上下文+用户问题交给DeepSeek，返回答案


## 环境搭建
### 1. 创建虚拟环境
```powershell
# 在项目根目录执行
python -m venv venv
# 激活虚拟环境
.\venv\Scripts\Activate.ps1
```

### 2. 安装依赖包
```powershell
pip install fastapi uvicorn langchain langchain-community langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers python-dotenv -i [https://mirrors.aliyun.com/pypi/simple/](https://mirrors.aliyun.com/pypi/simple/)
```

### 3. 环境变量配置
在项目根目录新建 `.env` 文件，填入DeepSeek密钥
```env
DEEPSEEK_API_KEY=你的deepseek密钥
DEEPSEEK_BASE_URL=[https://api.deepseek.com](https://api.deepseek.com)
```

## 运行项目
```powershell
# 激活虚拟环境后执行
python main.py
```
- 后端API文档： [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)（Swagger接口调试页面）
- 前端页面： [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

## 项目目录结构
```
sciflow/
├── main.py              # FastAPI入口、接口定义
├── rag_pipeline.py      # RAG核心流水线：加载PDF、向量库、检索逻辑
├── .env                 # 密钥配置文件（不要提交到git）
├── README.md
├── uploads/             # 上传PDF存放目录
├── vector_db/           # Chroma向量持久化存储目录
├── venv/                # Python虚拟环境
└── static/
    └── index.html       # 简易前端页面
```

## 功能说明
1. 上传PDF文件，后端自动解析、切片、存入向量数据库
2. 输入问题，后端检索文档相关片段，调用DeepSeek生成答案
3. 当文档无相关信息，模型返回：文档中没有找到相关内容，抑制大模型幻觉

## 注意事项
1. `.env` 文件包含API密钥，严禁上传到GitHub，添加到.gitignore
2. 首次运行会自动下载all-MiniLM-L6-v2嵌入模型，网络差可配置HF国内镜像
3. 向量数据保存在 vector_db，删除文件夹即可清空知识库

## 常见问题
1. HuggingFace模型下载超时：设置环境变量 `$env:HF_ENDPOINT="[https://hf-mirror.com](https://hf-mirror.com)"`
2. ImportError：不要把自己py文件命名和第三方库同名（如tavily.py）
3. DeprecationWarning：仅警告，不影响运行，是LangChain新版本拆分第三方包

