#!/bin/bash

# VLLM MCP Server Development Startup Script

set -e

# Default development values
TRANSPORT="stdio"
HOST="localhost"
PORT=8080
LOG_LEVEL="DEBUG"

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Development startup script for VLLM MCP Server"
            echo ""
            echo "Options:"
            echo "  --transport TRANSPORT    Transport type (stdio, http, sse) [default: stdio]"
            echo "  --host HOST              Host for HTTP/SSE transport [default: localhost]"
            echo "  --port PORT              Port for HTTP/SSE transport [default: 8080]"
            echo "  --log-level LEVEL        Log level [default: DEBUG]"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running the server"
    exit 1
fi

# Load environment variables
source .env

# Start the server with hot reload (if watchfiles is installed)
if uv pip show watchfiles > /dev/null 2>&1; then
    echo "Starting with hot reload..."
    export PYTHONPATH="src:$PYTHONPATH"
    exec uv run watchfiles --patterns="*.py" --recursive --signal=SIGTERM \
        python -m vllm_mcp.server \
        --transport $TRANSPORT \
        --host $HOST \
        --port $PORT \
        --log-level $LOG_LEVEL
else
    echo "Starting server (install watchfiles for hot reload: uv add --dev watchfiles)..."
    exec ./scripts/start.sh \
        --transport $TRANSPORT \
        --host $HOST \
        --port $PORT \
        --log-level $LOG_LEVEL
fi