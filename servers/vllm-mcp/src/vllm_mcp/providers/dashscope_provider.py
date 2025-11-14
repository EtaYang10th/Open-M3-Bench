"""Dashscope multimodal model provider."""

import base64
import mimetypes
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dashscope
from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import GenerationResponse

from ..models import (
    MultimodalRequest,
    MultimodalResponse,
    TextContent,
    ImageContent,
    FileContent,
)


class DashscopeProvider:
    """Dashscope multimodal model provider."""

    def __init__(self, api_key: str, supported_models: Optional[List[str]] = None):
        """Initialize Dashscope provider.

        Args:
            api_key: Dashscope API key
            supported_models: Optional list of supported models
        """
        dashscope.api_key = api_key
        self.supported_models = supported_models or [
            "qwen-vl-plus",
            "qwen-vl-max",
            "qwen-vl-chat",
            "qwen2-vl-7b-instruct",
            "qwen2-vl-72b-instruct",
        ]

    async def generate_response(
        self, request: MultimodalRequest
    ) -> MultimodalResponse:
        """Generate response from Dashscope multimodal model.

        Args:
            request: Multimodal request containing text, images, and files

        Returns:
            Multimodal response
        """
        try:
            messages = self._build_messages(request)

            response = MultiModalConversation.call(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            return self._parse_response(response)

        except Exception as e:
            raise Exception(f"Dashscope API error: {e}")

    def _build_messages(self, request: MultimodalRequest) -> List[Dict[str, Any]]:
        """Build Dashscope messages from multimodal request.

        Args:
            request: Multimodal request

        Returns:
            List of Dashscope message dictionaries
        """
        messages: List[Dict[str, Any]] = []

        # Add system message if present
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"text": request.system_prompt}]
            })

        # Build user message with multimodal content
        content: List[Dict[str, Any]] = []

        # Add text content
        for text_content in request.text_contents:
            content.append({"text": text_content.text})

        # Add image content
        for image_content in request.image_contents:
            image_data = self._prepare_image(image_content)
            content.append({"image": image_data})

        # Add file content (text files for Dashscope)
        for file_content in request.file_contents:
            if file_content.mime_type.startswith("text/"):
                text_data = self._prepare_text_file(file_content)
                content.append({"text": text_data})

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    def _prepare_image(self, image_content: ImageContent) -> str:
        """Prepare image data for Dashscope API.

        Args:
            image_content: Image content object

        Returns:
            Image URL or base64 data string
        """
        if image_content.url:
            return image_content.url
        elif image_content.image_path:
            return self._encode_image_from_path(image_content.image_path)
        elif image_content.base64_data:
            return f"data:{image_content.mime_type};base64,{image_content.base64_data}"
        else:
            raise ValueError("No valid image source provided")

    def _encode_image_from_path(self, image_path: str) -> str:
        """Encode image from file path to base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64 image data URL
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Invalid image file: {image_path}")

        with open(path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_data}"

    def _prepare_text_file(self, file_content: FileContent) -> str:
        """Prepare text file content for Dashscope API.

        Args:
            file_content: File content object

        Returns:
            Text content with filename
        """
        if file_content.text:
            return f"File: {file_content.filename}\n{file_content.text}"
        elif file_content.file_path:
            path = Path(file_content.file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_content.file_path}")

            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            return f"File: {file_content.filename}\n{content}"
        else:
            raise ValueError("No valid text file source provided")

    def _parse_response(self, response: GenerationResponse) -> MultimodalResponse:
        """Parse Dashscope response to multimodal response.

        Args:
            response: Dashscope generation response

        Returns:
            Multimodal response
        """
        if response.status_code != 200:
            return MultimodalResponse(
                model="",
                error=f"Dashscope API error: {response.message}"
            )

        output = response.output
        if not output or not output.choices:
            return MultimodalResponse(
                model=response.model or "",
                error="No response content"
            )

        choice = output.choices[0]
        message = choice.message

        # Extract text content
        text_contents = []
        if message.content:
            for content_item in message.content:
                if "text" in content_item:
                    text_contents.append(TextContent(text=content_item["text"]))

        return MultimodalResponse(
            text_contents=text_contents,
            model=response.model or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
        )

    def is_model_supported(self, model: str) -> bool:
        """Check if model is supported by provider.

        Args:
            model: Model name

        Returns:
            True if model is supported
        """
        return model in self.supported_models

    async def validate_request(self, request: MultimodalRequest) -> bool:
        """Validate multimodal request.

        Args:
            request: Multimodal request to validate

        Returns:
            True if request is valid
        """
        if not self.is_model_supported(request.model):
            return False

        # Check image count (Dashscope has limits)
        total_images = len(request.image_contents)
        if total_images > 10:  # Dashscope typically supports more images than OpenAI
            return False

        # Validate image formats
        for image_content in request.image_contents:
            if image_content.mime_type not in [
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp",
                "image/bmp"
            ]:
                return False

        return True

    async def stream_response(
        self, request: MultimodalRequest
    ) -> str:
        """Stream response from Dashscope multimodal model.

        Args:
            request: Multimodal request

        Yields:
            Chunks of response text
        """
        try:
            messages = self._build_messages(request)

            response = MultiModalConversation.call(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )

            for chunk in response:
                if chunk.output and chunk.output.choices:
                    choice = chunk.output.choices[0]
                    if choice.message and choice.message.content:
                        for content_item in choice.message.content:
                            if "text" in content_item:
                                yield content_item["text"]

        except Exception as e:
            yield f"Error: {e}"