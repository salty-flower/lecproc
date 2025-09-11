import hashlib
from io import BytesIO
from pathlib import Path
from typing import override

import anyio
from anyio import open_file
from litellm.utils import get_max_tokens
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseInputTextParam,
    WebSearchPreviewToolParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response_input_param import Message, ResponseInputParam
from pydantic import BaseModel, computed_field
from pydantic_settings import CliPositionalArg

from common_cli_settings import CommonCliSettings
from logs import create_progress

from .settings import deep_research_settings


class MyBatchRequest(BaseModel):
    custom_id: str
    body: ResponseCreateParamsBase
    url: str = "/v1/responses"
    method: str = "POST"


class Cli(CommonCliSettings):
    prompt_files: CliPositionalArg[list[str]]

    @computed_field
    @property
    async def system_prompt(self) -> str:
        async with await open_file(deep_research_settings.system_prompt_path, "r", encoding="utf-8") as f:
            return await f.read()

    async def craft_request(self, prompt: str) -> ResponseInputParam:
        return [
            Message(content=[ResponseInputTextParam(text=await self.system_prompt, type="input_text")], role="system"),
            Message(content=[ResponseInputTextParam(text=prompt, type="input_text")], role="user"),
        ]

    @override
    async def cli_cmd(self) -> None:
        prompts: list[str] = []
        for prompt_file in self.prompt_files:
            async with await open_file(Path(prompt_file), "r", encoding="utf-8") as f:
                prompts.append(await f.read())

        self.logger.info("Loaded %d prompts. Preparing the batch request...", len(prompts))
        batch_requests: list[MyBatchRequest] = []
        async for prompt, messages in ((prompt, await self.craft_request(prompt)) for prompt in prompts):
            batch_request = MyBatchRequest(
                custom_id=f"prompt-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}",
                body=ResponseCreateParamsBase(
                    model=deep_research_settings.model,
                    input=messages,
                    max_output_tokens=get_max_tokens(deep_research_settings.model),
                    temperature=1.0,
                    tools=[WebSearchPreviewToolParam(type="web_search_preview")],
                ),
            )
            batch_requests.append(batch_request)

        # Upload the file to openai
        self.logger.info("Uploading file to openai")
        client = AsyncOpenAI(api_key=deep_research_settings.openai_api_key)
        batch_file = await client.files.create(
            file=BytesIO("\n".join(request.model_dump_json() for request in batch_requests).encode("utf-8")),
            purpose="batch",
        )
        batch_job = await client.batches.create(
            completion_window="24h",
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
        )
        self.logger.info("Batch job created with ID %s", batch_job.id)

        # Busy waiting with Rich progress
        progress = create_progress()
        task_id = progress.add_task(f"Waiting for batch job {batch_job.id} to complete", total=1)
        while True:
            batch_job = await client.batches.retrieve(batch_job.id)
            # update progress
            progress.update(task_id, description=f"Waiting for batch job {batch_job.id}: {batch_job.status}")
            if batch_job.status == "completed":
                break
            await anyio.sleep(1)

        self.logger.info("Batch job %s completed", batch_job.id)


if __name__ == "__main__":
    _ = Cli.run_anyio()
