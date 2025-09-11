from io import BytesIO
from pathlib import Path
from typing import Any, cast, override

import anyio
import orjson
from anyio import open_file
from instructor.batch import BatchProcessor
from litellm.utils import get_max_tokens
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import computed_field
from pydantic_settings import CliPositionalArg

from common_cli_settings import CommonCliSettings
from logs import create_progress

from .settings import deep_research_settings


class Cli(CommonCliSettings):
    prompt_files: CliPositionalArg[list[str]]

    @computed_field
    @property
    async def system_prompt(self) -> str:
        async with await open_file(deep_research_settings.system_prompt_path, "r", encoding="utf-8") as f:
            return await f.read()


    async def craft_request(self, prompt: str) -> list[ChatCompletionUserMessageParam | ChatCompletionSystemMessageParam]:
        return [
            ChatCompletionSystemMessageParam(content=await self.system_prompt, role="system"),
            ChatCompletionUserMessageParam(content=prompt, role="user"),
        ]


    @override
    async def cli_cmd(self) -> None:
        prompts: list[str] = []
        for prompt_file in self.prompt_files:
            async with await open_file(Path(prompt_file), "r", encoding="utf-8") as f:
                prompts.append(await f.read())

        self.logger.info(
            "Loaded %d prompts. Preparing the batch request...", len(prompts)
        )
        bp = BatchProcessor(
            model=deep_research_settings.model,
            response_model=ChatCompletion,
        )
        jsonl_bytes = cast(
            "BytesIO",
            bp.create_batch_from_messages(
                messages_list=[cast("list[dict[str, Any]]", await self.craft_request(prompt)) for prompt in prompts],
                max_tokens=get_max_tokens(deep_research_settings.model),
                temperature=1.0,
            ),
        )

        # We don't really need a structured response.
        # Parse the response as a list of JSON object separated by newlines
        # For each object: remove the "body.response_format" key
        self.logger.info("Removing response_format from requests")
        parsed_requests = [
            cast("dict[str, Any]", orjson.loads(request))
            for request in jsonl_bytes.getvalue().decode("utf-8").splitlines()
        ]
        for request in parsed_requests:
            del request["body"]["response_format"]

        # Upload the file to openai
        self.logger.info("Uploading file to openai")
        client = AsyncOpenAI(api_key=deep_research_settings.openai_api_key)
        batch_file = await client.files.create(file=BytesIO(orjson.dumps(parsed_requests)), purpose="batch")
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
