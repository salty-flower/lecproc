from typing import Literal, TypedDict

from pydantic import BaseModel


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


class BookWideContext(BaseModel):
    title: str | None = None
    author: str | None = None
    description: str | None = None
    language: str | None = None
    publisher: str | None = None
    year: int | None = None

    table_of_contents: str | list[str] | None = None

    heading_preference: str | None = None
    notation_preference: str | None = None
    figure_preference: str | None = None


def compose_pdf_user_messages(
    pdf_file_name: str, base64_pdf: str, general_context: str | None = None
) -> list[SystemMessage | UserMessage]:
    draft: list[SystemMessage | UserMessage | None] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
                    You are an expert editor specializing in mathematical writing.
                    Your task is to convert PDF files to clean, readable Markdown.
                    Preserve headings, lists, tables and links.
                    Write formulas and symbols in LaTeX format.

                    Do not introduce any new information or opinions.
                    """,
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
