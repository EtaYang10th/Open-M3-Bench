"""Main MCP server for multimodal model interactions."""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions, Server

from .models import (
    MultimodalRequest,
    MultimodalResponse,
    TextContent,
    ImageContent,
    FileContent,
    ProviderConfig,
    MCPToolResult,
)
from .providers import OpenAIProvider, DashscopeProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultimodalMCPServer:
    """MCP server for multimodal model interactions."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the MCP server.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.providers = self._initialize_providers()
        self.server = FastMCP("vllm-mcp")
        self._setup_tools()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load server configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # Parse supported models from environment variables
        openai_models = os.getenv("OPENAI_SUPPORTED_MODELS", "gpt-4o,gpt-4o-mini,gpt-4-turbo,gpt-4-vision-preview").split(",")
        dashscope_models = os.getenv("DASHSCOPE_SUPPORTED_MODELS", "qwen-vl-plus,qwen-vl-max,qwen-vl-chat,qwen2-vl-7b-instruct,qwen2-vl-72b-instruct").split(",")

        # Default configuration
        return {
            "host": os.getenv("VLLM_MCP_HOST", "localhost"),
            "port": int(os.getenv("VLLM_MCP_PORT", "8080")),
            "transport": os.getenv("VLLM_MCP_TRANSPORT", "stdio"),
            "log_level": os.getenv("VLLM_MCP_LOG_LEVEL", "INFO"),
            "max_connections": 100,
            "request_timeout": 120,
            "providers": [
                {
                    "provider_type": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": os.getenv("OPENAI_BASE_URL"),
                    "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
                    "supported_models": [model.strip() for model in openai_models],
                    "max_tokens": 4000,
                    "temperature": 0.7
                },
                {
                    "provider_type": "dashscope",
                    "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
                    "default_model": os.getenv("DASHSCOPE_DEFAULT_MODEL", "qwen-vl-plus"),
                    "supported_models": [model.strip() for model in dashscope_models],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
            ]
        }

    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize model providers.

        Returns:
            Dictionary of provider instances
        """
        providers = {}

        for provider_config in self.config.get("providers", []):
            provider_type = provider_config.get("provider_type")

            if provider_type == "openai":
                if provider_config.get("api_key"):
                    providers["openai"] = OpenAIProvider(
                        api_key=provider_config["api_key"],
                        base_url=provider_config.get("base_url"),
                        supported_models=provider_config.get("supported_models")
                    )
                    default_model = provider_config.get("default_model", "gpt-4o")
                    logger.info(f"Initialized OpenAI provider with default model: {default_model}")

            elif provider_type == "dashscope":
                if provider_config.get("api_key"):
                    providers["dashscope"] = DashscopeProvider(
                        api_key=provider_config["api_key"],
                        supported_models=provider_config.get("supported_models")
                    )
                    default_model = provider_config.get("default_model", "qwen-vl-plus")
                    logger.info(f"Initialized Dashscope provider with default model: {default_model}")

        return providers

    def _setup_tools(self):
        """Setup MCP tools."""

        @self.server.tool()
        async def generate_multimodal_response(
            model: str,
            prompt: str,
            image_urls: Optional[List[str]] = None,
            file_paths: Optional[List[str]] = None,
            system_prompt: Optional[str] = None,
            max_tokens: Optional[int] = 1000,
            temperature: Optional[float] = 0.7,
            provider: Optional[str] = None
        ) -> str:
            """Generate response from multimodal model.

            Args:
                model: Model name to use
                prompt: Text prompt
                image_urls: Optional list of image URLs
                file_paths: Optional list of file paths
                system_prompt: Optional system prompt
                max_tokens: Maximum tokens to generate
                temperature: Generation temperature
                provider: Optional provider name (openai, dashscope)

            Returns:
                Generated response text
            """
            try:
                # Auto-detect provider if not specified
                if not provider:
                    if model.startswith("gpt"):
                        provider = "openai"
                    elif model.startswith("qwen"):
                        provider = "dashscope"
                    else:
                        provider = list(self.providers.keys())[0] if self.providers else None

                if not provider or provider not in self.providers:
                    return f"Error: Provider '{provider}' not available"

                # Build multimodal request
                text_contents = [TextContent(text=prompt)]
                image_contents = []
                file_contents = []

                # Add image content
                if image_urls:
                    for url in image_urls:
                        image_contents.append(ImageContent(
                            url=url,
                            mime_type="image/jpeg"  # Default, will be updated if needed
                        ))

                # Add file content
                if file_paths:
                    for file_path in file_paths:
                        path = Path(file_path)
                        if path.exists():
                            import mimetypes
                            mime_type, _ = mimetypes.guess_type(file_path)

                            if mime_type and mime_type.startswith("image/"):
                                image_contents.append(ImageContent(
                                    image_path=file_path,
                                    mime_type=mime_type
                                ))
                            elif mime_type and mime_type.startswith("text/"):
                                with open(path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                file_contents.append(FileContent(
                                    filename=path.name,
                                    text=content,
                                    mime_type=mime_type
                                ))

                request = MultimodalRequest(
                    model=model,
                    text_contents=text_contents,
                    image_contents=image_contents,
                    file_contents=file_contents,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Generate response (await provider)
                response = await self.providers[provider].generate_response(request)

                if response.error:
                    return f"Error: {response.error}"

                result = response.text
                if response.usage:
                    result += f"\n\n[Token usage: {response.usage}]"

                return result

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return f"Error: {str(e)}"

        @self.server.tool()
        def list_available_providers() -> str:
            """List available model providers and their configurations.

            Returns:
                JSON string of available providers and their models
            """
            providers_info = {}

            for provider_name, provider in self.providers.items():
                # Find the provider config to get default model
                provider_config = None
                for config in self.config.get("providers", []):
                    if config.get("provider_type") == provider_name:
                        provider_config = config
                        break

                if isinstance(provider, OpenAIProvider):
                    providers_info[provider_name] = {
                        "type": "openai",
                        "default_model": provider_config.get("default_model", "gpt-4o") if provider_config else "gpt-4o",
                        "supported_models": provider.supported_models,
                        "max_tokens": provider_config.get("max_tokens", 4000) if provider_config else 4000,
                        "temperature": provider_config.get("temperature", 0.7) if provider_config else 0.7
                    }
                elif isinstance(provider, DashscopeProvider):
                    providers_info[provider_name] = {
                        "type": "dashscope",
                        "default_model": provider_config.get("default_model", "qwen-vl-plus") if provider_config else "qwen-vl-plus",
                        "supported_models": provider.supported_models,
                        "max_tokens": provider_config.get("max_tokens", 4000) if provider_config else 4000,
                        "temperature": provider_config.get("temperature", 0.7) if provider_config else 0.7
                    }

            return json.dumps(providers_info, indent=2)

        @self.server.tool()
        async def validate_multimodal_request(
            model: str,
            image_count: int = 0,
            file_count: int = 0,
            provider: Optional[str] = None
        ) -> str:
            """Validate if a multimodal request is supported.

            Args:
                model: Model name to validate
                image_count: Number of images in request
                file_count: Number of files in request
                provider: Optional provider name

            Returns:
                Validation result
            """
            try:
                # Auto-detect provider if not specified
                if not provider:
                    if model.startswith("gpt"):
                        provider = "openai"
                    elif model.startswith("qwen"):
                        provider = "dashscope"
                    else:
                        provider = list(self.providers.keys())[0] if self.providers else None

                if not provider or provider not in self.providers:
                    return f"Error: Provider '{provider}' not available"

                # Create dummy request for validation
                request = MultimodalRequest(
                    model=model,
                    text_contents=[TextContent(text="test")],
                    image_contents=[
                        ImageContent(url="test.jpg", mime_type="image/jpeg")
                        for _ in range(image_count)
                    ],
                    file_contents=[
                        FileContent(filename="test.txt", text="test", mime_type="text/plain")
                        for _ in range(file_count)
                    ]
                )

                is_valid = await self.providers[provider].validate_request(request)

                if is_valid:
                    return f"Request is valid for provider '{provider}'"
                else:
                    return f"Request is invalid for provider '{provider}'"

            except Exception as e:
                return f"Error validating request: {str(e)}"

    def run(self, transport: str = "stdio", host: str = "localhost", port: int = 8080):
        """Run the MCP server.

        Args:
            transport: Transport type (stdio, http, sse)
            host: Host for HTTP/SSE transport
            port: Port for HTTP/SSE transport
        """
        logger.info(f"Starting MCP server with {transport} transport")

        if transport == "stdio":
            self.server.run()
        elif transport == "http":
            # Set environment variables for host and port before running
            import os
            os.environ["HOST"] = host
            os.environ["PORT"] = str(port)
            self.server.run(transport="streamable-http")
        elif transport == "sse":
            # Set environment variables for host and port before running
            import os
            os.environ["HOST"] = host
            os.environ["PORT"] = str(port)
            self.server.run(transport="sse")
        else:
            raise ValueError(f"Unsupported transport: {transport}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VLLM MCP Server")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--transport", type=str, default="stdio",
                       choices=["stdio", "http", "sse"],
                       help="Transport type")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host for HTTP/SSE transport")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for HTTP/SSE transport")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create and run server
    server = MultimodalMCPServer(args.config)
    server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()