"""Data models for multimodal interactions."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Text content model."""
    text: str = Field(..., description="Text content")


class ImageContent(BaseModel):
    """Image content model."""
    url: Optional[str] = Field(None, description="Image URL")
    image_path: Optional[str] = Field(None, description="Local image file path")
    base64_data: Optional[str] = Field(None, description="Base64 encoded image data")
    mime_type: str = Field(..., description="Image MIME type")
    description: Optional[str] = Field(None, description="Image description")

    def __init__(self, **data):
        super().__init__(**data)
        # Set default mime_type if not provided and we can guess it
        if not data.get("mime_type"):
            if data.get("image_path"):
                import mimetypes
                mime_type, _ = mimetypes.guess_type(data["image_path"])
                if mime_type:
                    self.mime_type = mime_type


class FileContent(BaseModel):
    """File content model."""
    filename: str = Field(..., description="File name")
    text: Optional[str] = Field(None, description="File text content")
    file_path: Optional[str] = Field(None, description="Local file path")
    mime_type: str = Field(..., description="File MIME type")
    size: Optional[int] = Field(None, description="File size in bytes")

    def __init__(self, **data):
        super().__init__(**data)
        # Set default mime_type if not provided
        if not data.get("mime_type"):
            if data.get("filename"):
                import mimetypes
                mime_type, _ = mimetypes.guess_type(data["filename"])
                if mime_type:
                    self.mime_type = mime_type


class MultimodalRequest(BaseModel):
    """Multimodal request model."""
    model: str = Field(..., description="Model name")
    text_contents: List[TextContent] = Field(default_factory=list, description="Text content list")
    image_contents: List[ImageContent] = Field(default_factory=list, description="Image content list")
    file_contents: List[FileContent] = Field(default_factory=list, description="File content list")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Generation temperature")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    stream: bool = Field(False, description="Whether to stream response")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra model parameters")

    def has_multimodal_content(self) -> bool:
        """Check if request contains multimodal content."""
        return len(self.image_contents) > 0 or len(self.file_contents) > 0


class MultimodalResponse(BaseModel):
    """Multimodal response model."""
    text_contents: List[TextContent] = Field(default_factory=list, description="Text content list")
    image_contents: List[ImageContent] = Field(default_factory=list, description="Generated image content")
    file_contents: List[FileContent] = Field(default_factory=list, description="Generated file content")
    model: str = Field(..., description="Model used")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    error: Optional[str] = Field(None, description="Error message if any")

    @property
    def text(self) -> str:
        """Get combined text content."""
        return "\n".join(content.text for content in self.text_contents)


class ProviderConfig(BaseModel):
    """Provider configuration model."""
    provider_type: str = Field(..., description="Provider type (openai, dashscope)")
    api_key: str = Field(..., description="API key")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    model_mapping: Dict[str, str] = Field(default_factory=dict, description="Model name mapping")
    default_model: str = Field(..., description="Default model to use")
    max_tokens: int = Field(4000, description="Maximum tokens")
    temperature: float = Field(0.7, description="Default temperature")
    timeout: int = Field(60, description="Request timeout in seconds")
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Extra configuration")


class ServerConfig(BaseModel):
    """Server configuration model."""
    host: str = Field("localhost", description="Server host")
    port: int = Field(8080, description="Server port")
    transport: str = Field("stdio", description="Transport type (stdio, http, sse)")
    log_level: str = Field("INFO", description="Log level")
    max_connections: int = Field(100, description="Maximum connections")
    request_timeout: int = Field(120, description="Request timeout in seconds")
    providers: List[ProviderConfig] = Field(..., description="List of provider configurations")


class MCPToolCall(BaseModel):
    """MCP tool call model."""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID")


class MCPToolResult(BaseModel):
    """MCP tool result model."""
    content: str = Field(..., description="Tool result content")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID")
    is_error: bool = Field(False, description="Whether result is an error")