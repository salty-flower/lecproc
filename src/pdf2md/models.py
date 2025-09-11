from typing import Literal, TypedDict

from anyio import open_file

from .settings import settings


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
    async with await open_file(settings.system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = await f.read()

    draft: list[SystemMessage | UserMessage | None] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
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
