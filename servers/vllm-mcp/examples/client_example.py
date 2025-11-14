#!/usr/bin/env python3
"""
Example client demonstrating how to use the VLLM MCP Server.
"""

import asyncio
import os
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    """Example client usage."""

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
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # Example 1: List available providers
            print("=== Example 1: List Available Providers ===")
            result = await session.call_tool("list_available_providers", {})
            print(result.content[0].text)
            print()

            # Example 2: Validate a request
            print("=== Example 2: Validate Request ===")
            result = await session.call_tool("validate_multimodal_request", {
                "model": "gpt-4o",
                "image_count": 2,
                "file_count": 1
            })
            print(result.content[0].text)
            print()

            # Example 3: Generate multimodal response (text only)
            print("=== Example 3: Generate Text Response ===")
            result = await session.call_tool("generate_multimodal_response", {
                "model": "gpt-4o",
                "prompt": "Explain what a Model Context Protocol server is in simple terms.",
                "max_tokens": 500,
                "temperature": 0.7
            })
            print(result.content[0].text)
            print()

            # Example 4: Generate multimodal response with image
            print("=== Example 4: Generate Multimodal Response ===")

            # Check if we have a sample image
            sample_image = "examples/sample_image.jpg"
            if Path(sample_image).exists():
                result = await session.call_tool("generate_multimodal_response", {
                    "model": "gpt-4o",
                    "prompt": "Describe what you see in this image.",
                    "image_urls": [f"file://{Path(sample_image).absolute()}"],
                    "max_tokens": 500
                })
                print(result.content[0].text)
            else:
                print(f"Sample image not found at {sample_image}")
                print("You can add an image file and update the path to test image processing")
            print()


if __name__ == "__main__":
    asyncio.run(main())