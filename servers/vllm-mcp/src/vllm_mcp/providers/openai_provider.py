"""OpenAI multimodal model provider."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from ..models import (
    MultimodalRequest,
    MultimodalResponse,
    TextContent,
    ImageContent,
    FileContent,
)


class OpenAIProvider:
    """OpenAI multimodal model provider."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, supported_models: Optional[List[str]] = None):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
            supported_models: Optional list of supported models
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.supported_models = supported_models or [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-vision-preview",
        ]

    async def generate_response(
        self, request: MultimodalRequest
    ) -> MultimodalResponse:
        """Generate response from OpenAI multimodal model.

        Args:
            request: Multimodal request containing text, images, and files

        Returns:
            Multimodal response
        """
        try:
            messages = self._build_messages(request)

            response = self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            return self._parse_response(response)

        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except Exception as e:
            raise Exception(f"Error generating response: {e}")

    def _build_messages(self, request: MultimodalRequest) -> List[ChatCompletionMessageParam]:
        """Build OpenAI messages from multimodal request.

        Args:
            request: Multimodal request

        Returns:
            List of OpenAI message parameters
        """
        messages: List[ChatCompletionMessageParam] = []

        # Add system message if present
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Build user message with multimodal content
        content: List[Dict[str, Any]] = []

        # Add text content
        for text_content in request.text_contents:
            content.append({"type": "text", "text": text_content.text})

        # Add image content
        for image_content in request.image_contents:
            image_data = self._prepare_image(image_content)
            content.append({
                "type": "image_url",
                "image_url": image_data
            })

        # Add file content (text files only for OpenAI)
        for file_content in request.file_contents:
            if file_content.mime_type.startswith("text/"):
                text_data = self._prepare_text_file(file_content)
                content.append({"type": "text", "text": text_data})

        messages.append({"role": "user", "content": content})

        return messages

    def _prepare_image(self, image_content: ImageContent) -> Dict[str, str]:
        """Prepare image data for OpenAI API.

        Args:
            image_content: Image content object

        Returns:
            Dictionary with image URL or base64 data
        """
        if image_content.url:
            return {"url": image_content.url}
        elif image_content.image_path:
            return self._encode_image_from_path(image_content.image_path)
        elif image_content.base64_data:
            return {
                "url": f"data:{image_content.mime_type};base64,{image_content.base64_data}"
            }
        else:
            raise ValueError("No valid image source provided")

    def _encode_image_from_path(self, image_path: str) -> Dict[str, str]:
        """Encode image from file path to base64.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with base64 image data
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Invalid image file: {image_path}")

        with open(path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")

        return {"url": f"data:{mime_type};base64,{base64_data}"}

    def _prepare_text_file(self, file_content: FileContent) -> str:
        """Prepare text file content for OpenAI API.

        Args:
            file_content: File content object

        Returns:
            Text content
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

    def _parse_response(self, response: ChatCompletion) -> MultimodalResponse:
        """Parse OpenAI response to multimodal response.

        Args:
            response: OpenAI chat completion response

        Returns:
            Multimodal response
        """
        choice = response.choices[0]
        message = choice.message

        return MultimodalResponse(
            text_contents=[TextContent(text=message.content or "")],
            model=response.model,
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

        # Check image count (OpenAI has limits)
        total_images = len(request.image_contents)
        if total_images > 5:  # OpenAI typically limits to 5 images
            return False

        # Validate image formats
        for image_content in request.image_contents:
            if image_content.mime_type not in [
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp"
            ]:
                return False

        return True