from pathlib import Path
from typing import TypeGuard

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)

from .prompt_loader import get_rendered_agent


def _is_system_message(message: AllMessageValues) -> TypeGuard[ChatCompletionSystemMessage]:
    return message["role"] == "system"


def _pdf_data_message(data_uri: str) -> ChatCompletionUserMessage:
    file_part: ChatCompletionFileObject = {"type": "file", "file": {"file_data": data_uri}}
    return {"role": "user", "content": [file_part]}


async def compose_pdf_user_messages(
    pdf_file_name: str, base64_pdf: str, general_context: str | None = None
) -> list[AllMessageValues]:
    # Use the new modular prompt system
    prompts_dir = Path(__file__).parent / "prompts"
    rendered_messages = await get_rendered_agent("drafter", prompts_dir)

    messages: list[AllMessageValues] = [msg for msg in rendered_messages if _is_system_message(msg)]

    if general_context:
        messages.append(
            {
                "role": "user",
                "content": (f"Here are some general context and my preferences:\n{general_context}"),
            }
        )

    messages.append({"role": "user", "content": f"Please process the PDF file named {pdf_file_name}:"})
    messages.append(_pdf_data_message(f"data:application/pdf;base64,{base64_pdf}"))

    return messages
