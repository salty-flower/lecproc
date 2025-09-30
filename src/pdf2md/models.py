from pathlib import Path
from typing import cast

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionFileObject,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)

from .prompt_loader import get_rendered_agent


def system_message(text: str) -> ChatCompletionSystemMessage:
    msg: ChatCompletionSystemMessage = {"role": "system", "content": text}
    return msg


def user_text_message(text: str) -> ChatCompletionUserMessage:
    msg: ChatCompletionUserMessage = {"role": "user", "content": text}
    return msg


def assistant_text_message(text: str) -> ChatCompletionAssistantMessage:
    msg: ChatCompletionAssistantMessage = {"role": "assistant", "content": text}
    return msg


def user_pdf_datauri_message(data_uri: str) -> ChatCompletionUserMessage:
    file_part: ChatCompletionFileObject = {"type": "file", "file": {"file_data": data_uri}}
    msg: ChatCompletionUserMessage = {"role": "user", "content": [file_part]}
    return msg


async def compose_pdf_user_messages(
    pdf_file_name: str, base64_pdf: str, general_context: str | None = None
) -> list[AllMessageValues]:
    # Use the new modular prompt system
    prompts_dir = Path(__file__).parent / "prompts"
    rendered_messages = await get_rendered_agent("drafter", prompts_dir)

    # Convert agent-rendered messages to system messages
    system_msgs: list[ChatCompletionSystemMessage] = []
    for msg in rendered_messages:
        if msg.get("role") == "system":
            content = cast(str, msg.get("content"))
            system_msgs.append({"role": "system", "content": content})

    messages: list[AllMessageValues | None] = [
        *system_msgs,
        user_text_message(
            f"""
            Here are some general context and my preferences:
            {general_context}
            """
        )
        if general_context
        else None,
        user_text_message(f"Please process the PDF file named {pdf_file_name}:"),
        user_pdf_datauri_message(f"data:application/pdf;base64,{base64_pdf}"),
    ]

    return cast(list[AllMessageValues], [m for m in messages if m is not None])
