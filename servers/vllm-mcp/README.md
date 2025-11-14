# VLLM MCP Server

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-0.5.0+-brightgreen.svg)](https://github.com/astral-sh/uv)

A Model Context Protocol (MCP) server that enables text models to call multimodal models. This server supports both OpenAI and Dashscope (Alibaba Cloud) multimodal models, allowing text-only models to process images and other media formats through standardized MCP tools.

**GitHub Repository**: https://github.com/StanleyChanH/vllm-mcp

## Features

- **Multi-Provider Support**: OpenAI GPT-4 Vision and Dashscope Qwen-VL models
- **Multiple Transport Options**: STDIO, HTTP, and Server-Sent Events (SSE)
- **Flexible Deployment**: Docker, Docker Compose, and local development
- **Easy Configuration**: JSON configuration files and environment variables
- **Comprehensive Tooling**: MCP tools for model interaction, validation, and provider management

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for OpenAI and/or Dashscope (阿里云)

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/StanleyChanH/vllm-mcp.git
   cd vllm-mcp
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   nano .env  # or use your preferred editor
   ```

3. **Configure API keys** (in `.env` file):
   ```bash
   # Dashscope (阿里云) - Required for basic functionality
   DASHSCOPE_API_KEY=sk-your-dashscope-api-key

   # OpenAI - Optional
   OPENAI_API_KEY=sk-your-openai-api-key
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

5. **Verify setup**:
   ```bash
   uv run python test_simple.py
   ```

### Running the Server

1. **Start the server** (STDIO transport - default):
   ```bash
   ./scripts/start.sh
   ```

2. **Start with HTTP transport**:
   ```bash
   ./scripts/start.sh --transport http --host 0.0.0.0 --port 8080
   ```

3. **Development mode with hot reload**:
   ```bash
   ./scripts/start-dev.sh
   ```

### Testing & Verification

1. **List available models**:
   ```bash
   uv run python examples/list_models.py
   ```

2. **Run basic tests**:
   ```bash
   uv run python test_simple.py
   ```

3. **Test MCP tools**:
   ```bash
   uv run python examples/client_example.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   # Create .env file with your API keys
   cp .env.example .env

   # Start the service
   docker-compose up -d
   ```

2. **Build manually**:
   ```bash
   docker build -t vllm-mcp .
   docker run -p 8080:8080 --env-file .env vllm-mcp
   ```

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_SUPPORTED_MODELS=gpt-4o,gpt-4o-mini,gpt-4-turbo,gpt-4-vision-preview

# Dashscope Configuration
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_DEFAULT_MODEL=qwen-vl-plus
DASHSCOPE_SUPPORTED_MODELS=qwen-vl-plus,qwen-vl-max,qwen-vl-chat,qwen2-vl-7b-instruct,qwen2-vl-72b-instruct

# Server Configuration (optional)
VLLM_MCP_HOST=localhost
VLLM_MCP_PORT=8080
VLLM_MCP_TRANSPORT=stdio
VLLM_MCP_LOG_LEVEL=INFO
```

### Configuration File

Create a `config.json` file:

```json
{
  "host": "localhost",
  "port": 8080,
  "transport": "stdio",
  "log_level": "INFO",
  "providers": [
    {
      "provider_type": "openai",
      "api_key": "${OPENAI_API_KEY}",
      "base_url": "${OPENAI_BASE_URL}",
      "default_model": "gpt-4o",
      "max_tokens": 4000,
      "temperature": 0.7
    },
    {
      "provider_type": "dashscope",
      "api_key": "${DASHSCOPE_API_KEY}",
      "default_model": "qwen-vl-plus",
      "max_tokens": 4000,
      "temperature": 0.7
    }
  ]
}
```

## MCP Tools

The server provides the following MCP tools:

### `generate_multimodal_response`

Generate responses from multimodal models.

**Parameters:**
- `model` (string): Model name to use
- `prompt` (string): Text prompt
- `image_urls` (array, optional): List of image URLs
- `file_paths` (array, optional): List of file paths
- `system_prompt` (string, optional): System prompt
- `max_tokens` (integer, optional): Maximum tokens to generate
- `temperature` (number, optional): Generation temperature
- `provider` (string, optional): Provider name (auto-detected if not specified)

**Example:**
```python
result = await session.call_tool("generate_multimodal_response", {
    "model": "gpt-4o",
    "prompt": "Describe this image",
    "image_urls": ["https://example.com/image.jpg"],
    "max_tokens": 500
})
```

### `list_available_providers`

List available model providers and their supported models.

**Example:**
```python
result = await session.call_tool("list_available_providers", {})
```

### `validate_multimodal_request`

Validate if a multimodal request is supported by the specified provider.

**Parameters:**
- `model` (string): Model name to validate
- `image_count` (integer, optional): Number of images
- `file_count` (integer, optional): Number of files
- `provider` (string, optional): Provider name

## Supported Models

### OpenAI
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4-vision-preview`

### Dashscope
- `qwen-vl-plus`
- `qwen-vl-max`
- `qwen-vl-chat`
- `qwen2-vl-7b-instruct`
- `qwen2-vl-72b-instruct`

## Model Selection

### Using Environment Variables

You can configure default models and supported models through environment variables:

```bash
# OpenAI
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_SUPPORTED_MODELS=gpt-4o,gpt-4o-mini,gpt-4-turbo

# Dashscope
DASHSCOPE_DEFAULT_MODEL=qwen-vl-plus
DASHSCOPE_SUPPORTED_MODELS=qwen-vl-plus,qwen-vl-max
```

### Listing Available Models

Use the `list_available_providers` tool to see all available models:

```python
result = await session.call_tool("list_available_providers", {})
print(result.content[0].text)
```

### Model Selection Examples

```python
# Use specific OpenAI model
result = await session.call_tool("generate_multimodal_response", {
    "model": "gpt-4o-mini",  # Specify exact model
    "prompt": "Analyze this image",
    "image_urls": ["https://example.com/image.jpg"]
})

# Use specific Dashscope model
result = await session.call_tool("generate_multimodal_response", {
    "model": "qwen-vl-max",  # Specify exact model
    "prompt": "Describe what you see",
    "image_urls": ["https://example.com/image.jpg"]
})

# Auto-detect provider based on model name
# OpenAI models (gpt-*) will use OpenAI provider
# Dashscope models (qwen-*) will use Dashscope provider
```

### Model Configuration File

You can also configure models in `config.json`:

```json
{
  "providers": [
    {
      "provider_type": "openai",
      "api_key": "${OPENAI_API_KEY}",
      "default_model": "gpt-4o-mini",
      "supported_models": ["gpt-4o-mini", "gpt-4-turbo"],
      "max_tokens": 4000,
      "temperature": 0.7
    },
    {
      "provider_type": "dashscope",
      "api_key": "${DASHSCOPE_API_KEY}",
      "default_model": "qwen-vl-max",
      "supported_models": ["qwen-vl-plus", "qwen-vl-max"],
      "max_tokens": 4000,
      "temperature": 0.7
    }
  ]
}
```

## Client Integration

### Python Client

```python
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "vllm_mcp.server"],
        env={"PYTHONPATH": "src"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Generate multimodal response
            result = await session.call_tool("generate_multimodal_response", {
                "model": "gpt-4o",
                "prompt": "Analyze this image",
                "image_urls": ["https://example.com/image.jpg"]
            })

            print(result.content[0].text)

asyncio.run(main())
```

### MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "vllm-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "vllm_mcp.server"],
      "env": {
        "PYTHONPATH": "src",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "DASHSCOPE_API_KEY": "${DASHSCOPE_API_KEY}"
      }
    }
  }
}
```

## Development

### Project Structure

```
vllm-mcp/
├── src/vllm_mcp/
│   ├── __init__.py
│   ├── server.py          # Main MCP server
│   ├── models.py          # Data models
│   └── providers/
│       ├── __init__.py
│       ├── openai_provider.py
│       └── dashscope_provider.py
├── scripts/
│   ├── start.sh           # Production startup
│   └── start-dev.sh       # Development startup
├── examples/
│   ├── client_example.py  # Example client
│   └── mcp_client_config.json
├── docker-compose.yml
├── Dockerfile
├── config.json
└── README.md
```

### Adding New Providers

1. Create a new provider class in `src/vllm_mcp/providers/`
2. Implement the required methods:
   - `generate_response()`
   - `is_model_supported()`
   - `validate_request()`
3. Register the provider in `src/vllm_mcp/server.py`
4. Update configuration schema

### Running Tests

```bash
# Install development dependencies
uv add --dev pytest pytest-asyncio

# Run tests
uv run pytest
```

## Deployment Options

### STDIO Transport (Default)
Best for MCP client integrations and local development.

```bash
vllm-mcp --transport stdio
```

### HTTP Transport
Suitable for web service deployments.

```bash
vllm-mcp --transport http --host 0.0.0.0 --port 8080
```

### SSE Transport
For real-time streaming responses.

```bash
vllm-mcp --transport sse --host 0.0.0.0 --port 8080
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'vllm_mcp'**
   ```bash
   # Make sure you're in the project root and run:
   uv sync
   export PYTHONPATH="src:$PYTHONPATH"
   ```

2. **API Key Not Found**
   ```bash
   # Ensure your .env file is properly configured:
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. **Dashscope API Errors**
   - Verify your API key is valid and active
   - Check if you have sufficient quota
   - Ensure network connectivity to Dashscope services

4. **Server Startup Issues**
   ```bash
   # Check for port conflicts:
   lsof -i :8080

   # Try a different port:
   ./scripts/start.sh --port 8081
   ```

5. **Docker Issues**
   ```bash
   # Rebuild Docker image:
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Debug Mode

Enable debug logging for troubleshooting:
```bash
./scripts/start.sh --log-level DEBUG
```

### Getting Help

- Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions
- Run `uv run python test_simple.py` to verify basic functionality
- Review logs for error messages and warnings

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- Issues: [GitHub Issues](https://github.com/StanleyChanH/vllm-mcp/issues)
- Documentation: [Wiki](https://github.com/StanleyChanH/vllm-mcp/wiki)

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [OpenAI](https://openai.com/)
- [Dashscope](https://dashscope.aliyun.com/)