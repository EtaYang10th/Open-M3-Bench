# 🔑 模型密钥配置指南

## 推荐方式：使用 .env 文件

### 1. 编辑 .env 文件
```bash
# 使用你喜欢的编辑器打开 .env 文件
nano .env
# 或
vim .env
```

### 2. 配置 OpenAI (可选)
```bash
# 替换为你的真实 OpenAI API 密钥
OPENAI_API_KEY=sk-your-real-openai-api-key-here

# 可选：自定义 OpenAI API 基础 URL
OPENAI_BASE_URL=https://api.openai.com/v1

# 可选：选择默认模型
OPENAI_DEFAULT_MODEL=gpt-4o

# 可选：配置支持的模型列表
OPENAI_SUPPORTED_MODELS=gpt-4o,gpt-4o-mini,gpt-4-turbo
```

### 3. 配置 Dashscope (阿里云)
```bash
# 替换为你的真实 Dashscope API 密钥
DASHSCOPE_API_KEY=sk-your-real-dashscope-api-key-here

# 可选：选择默认模型
DASHSCOPE_DEFAULT_MODEL=qwen-vl-plus

# 可选：配置支持的模型列表
DASHSCOPE_SUPPORTED_MODELS=qwen-vl-plus,qwen-vl-max
```

## 其他配置方式

### 方式 2：命令行环境变量
```bash
# OpenAI
export OPENAI_API_KEY="sk-your-real-openai-api-key"

# Dashscope
export DASHSCOPE_API_KEY="sk-your-real-dashscope-api-key"

# 启动服务器
./scripts/start.sh
```

### 方式 3：Docker Compose
```bash
# 在 docker-compose.yml 中设置环境变量
# 或使用 .env 文件（Docker 会自动读取）

docker-compose up -d
```

## 🔍 如何获取 API 密钥

### OpenAI API 密钥
1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 登录或注册账户
3. 进入 API Keys 页面
4. 创建新的 API 密钥
5. 复制密钥并填入 `.env` 文件

### Dashscope API 密钥
1. 访问 [阿里云 Dashscope](https://dashscope.aliyun.com/)
2. 登录阿里云账户
3. 进入控制台 -> API-KEY 管理
4. 创建新的 API-KEY
5. 复制密钥并填入 `.env` 文件

## 🧪 验证配置

配置完成后，运行测试验证：
```bash
# 运行设置测试
uv run python test_setup.py

# 列出可用模型
uv run python examples/list_models.py
```

## 🔒 隐私保护

- ✅ `.env` 文件已被 `.gitignore` 忽略，不会被提交到 Git
- ✅ 你的 API 密钥是安全的
- ✅ 可以在 `.env.example` 中查看配置模板