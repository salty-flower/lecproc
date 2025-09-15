from pathlib import Path
from typing import Literal, TypedDict

from .prompt_loader import get_rendered_agent


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class FileInner(TypedDict):
    file_data: str


class FilePart(TypedDict):
    type: Literal["file"]
    file: FileInner


ContentPart = TextPart | FilePart


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: list[ContentPart]


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[ContentPart]


async def compose_pdf_user_messages(
    pdf_file_name: str, base64_pdf: str, general_context: str | None = None
) -> list[SystemMessage | UserMessage]:
    # Use the new modular prompt system
    prompts_dir = Path(__file__).parent / "prompts"
    rendered_messages = await get_rendered_agent("drafter", prompts_dir)

    # Convert rendered messages to the expected format
    system_parts: list[ContentPart] = [
        {"type": "text", "text": msg["content"]} for msg in rendered_messages if msg["role"] == "system"
    ]

    system_message: SystemMessage = {
        "role": "system",
        "content": system_parts,
    }

    draft: list[SystemMessage | UserMessage | None] = [
        system_message,
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    Here are some general context and my preferences:
                    {general_context}
                    """,
                },
            ],
        }
        if general_context
        else None,
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please process the PDF file named {pdf_file_name}:",
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    },
                },
            ],
        },
    ]

    return [f for f in draft if f is not None]
