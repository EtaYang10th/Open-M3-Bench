#!/usr/bin/env python3
"""
Example script to demonstrate how to list available models and providers.
"""

import asyncio
import os
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    """Example showing how to list available providers and models."""

    print("🔍 VLLM MCP Server - Model Listing Example")
    print("=" * 50)

    # Server parameters for stdio connection
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "python",
            "-m",
            "vllm_mcp.server",
            "--transport",
            "stdio",
            "--log-level",
            "INFO"
        ],
        env={
            **os.environ,
            "PYTHONPATH": "src",
        }
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("🛠️  Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # List available providers and models
            print("🤖 Available providers and models:")
            try:
                result = await session.call_tool("list_available_providers", {})
                providers_info = result.content[0].text

                import json
                providers = json.loads(providers_info)

                for provider_name, info in providers.items():
                    print(f"\n📋 Provider: {provider_name} ({info['type']})")
                    print(f"   Default model: {info['default_model']}")
                    print(f"   Max tokens: {info['max_tokens']}")
                    print(f"   Temperature: {info['temperature']}")
                    print(f"   Supported models:")
                    for model in info['supported_models']:
                        print(f"     - {model}")

            except Exception as e:
                print(f"❌ Error listing providers: {e}")

            print()

            # Show model selection examples
            print("📝 Model selection examples:")
            print("\n1. Using OpenAI GPT-4o:")
            print("   Call 'generate_multimodal_response' with:")
            print("   - model: 'gpt-4o'")
            print("   - provider: 'openai' (optional, auto-detected)")

            print("\n2. Using Dashscope Qwen-VL-Plus:")
            print("   Call 'generate_multimodal_response' with:")
            print("   - model: 'qwen-vl-plus'")
            print("   - provider: 'dashscope' (optional, auto-detected)")

            print("\n3. Text-only example:")
            print("""
            result = await session.call_tool("generate_multimodal_response", {
                "model": "gpt-4o",
                "prompt": "Explain machine learning",
                "max_tokens": 500
            })
            """)

            print("\n4. Multimodal example with image:")
            print("""
            result = await session.call_tool("generate_multimodal_response", {
                "model": "qwen-vl-plus",
                "prompt": "Describe this image",
                "image_urls": ["https://example.com/image.jpg"],
                "max_tokens": 500
            })
            """)

            print("\n💡 Configuration:")
            print("You can modify the default models and supported models in:")
            print("- .env file (environment variables)")
            print("- config.json file")
            print("\nEnvironment variables:")
            print("- OPENAI_DEFAULT_MODEL")
            print("- OPENAI_SUPPORTED_MODELS")
            print("- DASHSCOPE_DEFAULT_MODEL")
            print("- DASHSCOPE_SUPPORTED_MODELS")


if __name__ == "__main__":
    asyncio.run(main())