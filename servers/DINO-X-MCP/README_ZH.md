# DINO-X MCP

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![npm version](https://img.shields.io/npm/v/@deepdataspace/dinox-mcp.svg)](https://www.npmjs.com/package/@deepdataspace/dinox-mcp) [![npm downloads](https://img.shields.io/npm/dm/@deepdataspace/dinox-mcp.svg)](https://www.npmjs.com/package/@deepdataspace/dinox-mcp) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/IDEA-Research/DINO-X-MCP/pulls) [![MCP Badge](https://lobehub.com/badge/mcp/idea-research-dino-x-mcp)](https://lobehub.com/mcp/idea-research-dino-x-mcp) [![GitHub stars](https://img.shields.io/github/stars/IDEA-Research/DINO-X-MCP.svg)](https://github.com/IDEA-Research/DINO-X-MCP/stargazers)

[English](README.md) | **中文**

<p align="center">

DINO-X 官方 MCP 服务器, 基于通用视觉检测模型 DINO-X 和 Grounding DINO，给大模型提供细粒度的目标检测、图像理解、图片分析能力。

<p align="center">
  <video width="800" controls>
    <source src="https://dds-frontend.oss-cn-shenzhen.aliyuncs.com/dinox-mcp/dinox-mcp-en-overveiw.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

</p>

## 为什么需要 DINO-X MCP？

- 精细理解：支持全图检测、定向检测与区域描述。

- 结构化输出：可直接获取类别、数量、位置、属性，助力视觉问答与多步推理。

- 组合易用：可与其他 MCP Server 搭配，构建端到端视觉智能体与自动化流程。

## 传输模式

DINO-X MCP 支持两种传输模式:

| 特性 | STDIO（默认） | Streamable HTTP |
|:--|:--|:--|
| 运行环境 | 本地 | 本地或云端 |
| 通信方式 | 标准输入/输出 | `HTTP`（流式响应） |
| 输入限制 | 支持 `file://` 与 `https://` | 仅支持 `https://` |
| 可视化 | 支持（生成本地标注图片） | 暂不支持 |

## 快速接入


### 1. 准备 MCP 客户端

首先, 你需要在本地安装一个支持 MCP 协议的客户端，例如：

- [Cursor](https://www.cursor.com/)
- [WindSurf](https://windsurf.com/)
- [Trae](https://www.trae.ai/)
- [Cherry Studio](https://www.cherry-ai.com/)

### 2. 获取 API Key 

前往 [DINO-X 官网](https://cloud.deepdataspace.com/request_api) 注册账号并申请 API Token（新用户有免费额度）。

### 3. 配置 MCP

#### 方式 A：使用官方托管的 Streamable HTTP（推荐）

在客户端配置文件中添加，替换自己的 API Key：

```json
{
  "mcpServers": {
    "dinox-mcp": {
      "url": "https://mcp.deepdataspace.com/mcp?key=your-api-key"
    }
  }
}

```

#### 方式 B：使用 NPM 包本地运行（STDIO）

首先，需要本地安装 `Node.js` 环境，可以从 [Node.js 官网](https://nodejs.org/) 下载安装包，或使用命令安装：

```bash
# MacOS 或 Linux 系统
# 1. 安装 nvm (Node 版本管理器)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# 或
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# 2. 将以下配置添加到你的终端配置文件中（~/.bash_profile、~/.zshrc、~/.profile 或 ~/.bashrc）
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  

# 3. 在当前终端中激活 nvm
# bash 用户运行：
source ~/.bashrc
# zsh 用户运行：
source ~/.zshrc   

# 4. 验证 nvm 是否安装成功
command -v nvm

# 5. 安装并使用 Node.js LTS 版本
nvm install --lts
nvm use --lts

# Windows 系统
winget install OpenJS.NodeJS.LTS
# 或使用 PowerShell (管理员权限)
iwr -useb https://raw.githubusercontent.com/chocolatey/chocolatey/master/chocolateyInstall/InstallChocolatey.ps1 | iex
choco install nodejs-lts -y
```

然后，在你的 MCP 客户端的配置文件中加入配置：

```json
{
  "mcpServers": {
    "dinox-mcp": {
      "command": "npx",
      "args": ["-y", "@deepdataspace/dinox-mcp"],
      "env": {
        "DINOX_API_KEY": "you-api-key-here",
        "IMAGE_STORAGE_DIRECTORY": "/path/to/your/image/directory"
      }
    }
  }
}
```

注意：获取 API Key 后，把上面配置中的 `you-api-key-here` 替换成真正的密钥。

#### 方式 C：克隆源码本地运行

首先，需要本地安装 `Node.js` 环境（参考方式二）

```bash
# 下载源码
git clone https://github.com/IDEA-Research/DINO-X-MCP.git
cd DINO-X-MCP

# 安装依赖
npm install

# 编译
npm run build
```

MCP 客户端中配置：

```json
{
  "mcpServers": {
    "dinox-mcp": {
      "command": "node",
      "args": ["/path/to/DINO-X-MCP/build/index.js"],
      "env": {
        "DINOX_API_KEY": "you-api-key-here",
        "IMAGE_STORAGE_DIRECTORY": "/path/to/your/image/directory"
      }
    }
  }
}
```

## 启动参数与环境变量

- 常用启动参数
  - `--http`：以 Streamable HTTP 方式启动；否则默认 STDIO
  - `--stdio`：强制以 STDIO 启动
  - `--dinox-api-key=...`：指定 API Key
  - `--enable-client-key`：允许客户端通过 URL `?key=` 传入 API Key（Streamable HTTP 模式）
  - `--port=8080`：HTTP 端口（默认 3020）

- 环境变量
  - `DINOX_API_KEY`（必需/条件必需）：DINO-X 平台鉴权密钥
  - `IMAGE_STORAGE_DIRECTORY`（可选，STDIO 模式）：可视化图片输出目录
  - `AUTH_TOKEN`（可选，Streamable HTTP 模式）：若设置，则需客户端请求头携带 `Authorization: Bearer <token>`

示例：
```bash
# STDIO（本地）
node build/index.js --dinox-api-key=your-api-key

# Streamable HTTP（服务端提供统一的 API Key）
node build/index.js --http --dinox-api-key=your-api-key

# Streamable HTTP（自定义端口）
node build/index.js --http --dinox-api-key=your-api-key --port=8080

# Streamable HTTP（强制客户端提供 API Key）
node build/index.js --http --enable-client-key

# 此时，客户端需要通过 ?key= 传入
{
  "mcpServers": {
    "dinox-mcp": {
      "url": "http://localhost:3020/mcp?key=your-api-key"
    }
  }
}

# Streamable HTTP（启用 AUTH_TOKEN）
AUTH_TOKEN=my-token node build/index.js --http --enable-client-key

# 此时，客户端可以借助 supergateway 传入 Authorization: Bearer <token>
{
  "mcpServers": {
    "dinox-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "supergateway",
        "--streamableHttp",
        "http://localhost:3020/mcp?key=your-api-key",
        "--oauth2Bearer",
        "my-token"
      ]
    }
  }
}

```

## 🛠️ 功能支持

| 能力 | 工具 ID | 传输模式 | 输入 | 输出 |
|:--|:--|:--|:--|:--|
| 全图万物检测 | `detect-all-objects` | STDIO/HTTP | 图片链接 | 类别 + bbox +（可选）描述 |
| 指定目标检测 | `detect-objects-by-text` | STDIO/HTTP | 图片链接 + 英文名词（用点号分隔多类，如 `person.car`） | 指定对象的 bbox +（可选）描述 |
| 人体姿态检测 | `detect-human-pose-keypoints` | STDIO/HTTP | 图片链接 | 17 个关键点 + bbox +（可选）描述 |
| 检测结果可视化 | `visualize-detection-result` | 仅 STDIO | 图片链接 + 检测结果数组 | 生成本地标注图片路径 |

## 应用案例

| 场景 | 输入 | 输出 |
|---------|---------|---------|
| **检测定位** | **💬 提示词：**<br>`帮我框选森林里的`<br>`着火范围并可视化`<br><br>**🖼️ 输入图片：**<br>![1-1](https://github.com/user-attachments/assets/b5401c20-b7f2-4a8e-bc24-4eca54c48a92) | ![1-2](https://github.com/user-attachments/assets/8bc3c9fe-5a5a-4f10-af0a-552b797a00fc) |
| **物体计数** | **💬 提示词：**<br>`请帮我分析这张`<br>`仓库图片，检测其中`<br>`的所有纸箱，统计`<br>`总数量`<br><br>**🖼️ 输入图片：**<br>![2-1](https://github.com/user-attachments/assets/9d500523-4094-43fe-a6e5-5a714075dfd8) | <img width="1276" alt="2-2" src="https://github.com/user-attachments/assets/3f18ef8c-5e89-45fc-bd0f-f23381304272" /> |
| **特征检测** | **💬 提示词：**<br>`找到图中所有红色`<br>`的车，并可视化展示`<br><br>**🖼️ 输入图片：**<br>![4-1](https://github.com/user-attachments/assets/3a5b0776-a932-447f-bc81-42c0536381e8)|![4-2](https://github.com/user-attachments/assets/410c2120-8c86-4f90-9ce1-c34fb3b1781a)|
| **属性推理** | **💬 提示词：**<br>`找到图中最高的人，`<br>`并描述他的着装，`<br>`用 Canvas 可视化`<br><br>**🖼️ 输入图片：**<br>![5-1](https://github.com/user-attachments/assets/9ba4b77e-6d00-4257-9bdb-079a53ce4ca4) | ![5-2](https://github.com/user-attachments/assets/ef0ce3e1-1dd2-4bd7-a145-e1623c297911) |
| **全图检测** | **💬 提示词：**<br>`找到图中维生素C`<br>`含量最高的水果`<br><br>**🖼️ 输入图片：**<br>![6-1](https://github.com/user-attachments/assets/9cf9f475-6015-47d0-917e-5004a104d777)| ![6-3](https://github.com/user-attachments/assets/17466bc2-4b9a-4a74-9456-05c253b0b388)<br><br>*Answer: Kiwi fruit (93mg/100g)* |
| **姿态分析** | **💬 提示词：**<br>`请分析这是什么`<br>`瑜伽姿势，并用`<br>`canvas 显示关键点`<br><br>**🖼️ 输入图片：**<br>![3-1](https://github.com/user-attachments/assets/2b2a6b5a-939b-4c37-8f40-900ae921b07a) |![3-3](https://github.com/user-attachments/assets/41636813-aaf7-4ad0-a5c3-1a5cbe67c022)|


## 常见问题

- 支持哪些图片来源？
  - STDIO：`file://` 与 `https://`
  - Streamable HTTP：仅 `https://`
- 支持哪些图片格式?
  - Jpg, Jpeg, Webp, Png

## 开发与调试

如果你要修改这个 MCP 服务，可以开启监听模式，代码改动会自动重新编译：

```bash
npm run watch
```

遇到问题时，可以用官方调试工具排查：

```bash
npm run inspector
```

## 协议

Apache License 2.0
