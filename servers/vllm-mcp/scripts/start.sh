#!/bin/bash

# VLLM MCP Server Startup Script

set -e

# Default values
TRANSPORT="stdio"
HOST="localhost"
PORT="8080"
LOG_LEVEL="INFO"
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --transport)
            TRANSPORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --transport TRANSPORT    Transport type (stdio, http, sse) [default: stdio]"
            echo "  --host HOST              Host for HTTP/SSE transport [default: localhost]"
            echo "  --port PORT              Port for HTTP/SSE transport [default: 8080]"
            echo "  --log-level LEVEL        Log level (DEBUG, INFO, WARNING, ERROR) [default: INFO]"
            echo "  --config FILE            Configuration file path"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Start with stdio transport"
            echo "  $0 --transport http --host 0.0.0.0   # Start with HTTP transport"
            echo "  $0 --transport sse --port 3000       # Start with SSE transport"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Check environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Warning: Neither OPENAI_API_KEY nor DASHSCOPE_API_KEY is set"
    echo "Please set at least one of these environment variables"
    echo ""
    echo "You can create a .env file from .env.example:"
    echo "  cp .env.example .env"
    echo "Then edit .env with your API keys"
    echo ""
fi

# Build command arguments
CMD_ARGS="vllm-mcp"
CMD_ARGS="$CMD_ARGS --transport $TRANSPORT"
CMD_ARGS="$CMD_ARGS --host $HOST"
CMD_ARGS="$CMD_ARGS --port $PORT"
CMD_ARGS="$CMD_ARGS --log-level $LOG_LEVEL"

if [ -n "$CONFIG_FILE" ]; then
    CMD_ARGS="$CMD_ARGS --config $CONFIG_FILE"
fi

echo "Starting VLLM MCP Server..."
echo "Transport: $TRANSPORT"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config File: $CONFIG_FILE"
fi
echo ""

# Start the server
exec $CMD_ARGS