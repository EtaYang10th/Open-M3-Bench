# Multi-stage build for VLLM MCP Server
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN uv venv /opt/venv
RUN . /opt/venv/bin/activate && uv pip install -e .

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy application
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config.json /app/config.json
COPY --from=builder /app/.env.example /app/.env.example

# Create non-root user
RUN useradd --create-home --shell /bin/bash vllm-mcp
RUN chown -R vllm-mcp:vllm-mcp /app
RUN chown -R vllm-mcp:vllm-mcp /opt/venv

USER vllm-mcp

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV VLLM_MCP_HOST=0.0.0.0
ENV VLLM_MCP_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set working directory
WORKDIR /app

# Default command
CMD ["vllm-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]