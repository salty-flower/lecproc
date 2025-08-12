from typing import Literal, TypedDict


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class FileInner(TypedDict):
    file_data: str


class FilePart(TypedDict):
    type: Literal["file"]
    file: FileInner


ContentPart = TextPart | FilePart


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[ContentPart]


def compose_pdf_user_messages(base64_pdf: str) -> list[UserMessage]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Convert this PDF to clean, readable Markdown. "
                        "Preserve headings, lists, tables and links. "
                        "Write formulas and symbols in LaTeX format."
                    ),
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    },
                },
            ],
        }
    ]
