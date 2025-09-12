import contextlib
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, cast, override

import anyio
import orjson
from anyio import open_file
from litellm.utils import get_max_tokens
from openai import AsyncOpenAI
from openai.types import Batch
from openai.types.responses import (
    Response,
    ResponseFunctionWebSearch,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    WebSearchPreviewToolParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_input_param import Message, ResponseInputParam
from pydantic import BaseModel, ValidationError, computed_field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand
from rich.console import Console
from rich.panel import Panel

from common_cli_settings import CommonCliSettings
from logs import create_progress

from .settings import deep_research_settings


class MyBatchRequest(BaseModel):
    custom_id: str
    body: ResponseCreateParamsBase
    url: str = "/v1/responses"
    method: str = "POST"


class BatchResponseWrapper(BaseModel):
    status_code: int
    request_id: str
    body: Response


class BatchResponse(BaseModel):
    id: str
    custom_id: str
    response: BatchResponseWrapper | None = None
    error: dict[str, Any] | None = None


class Retrieve(CommonCliSettings):
    batch_job_id: CliPositionalArg[str]

    @computed_field
    @property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=deep_research_settings.openai_api_key)

    def fix_response_data(self, response_output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Fix web_search_call items with broken action types
        new_response_output: list[dict[str, Any]] = []
        for obj in response_output:
            if obj.get("type") != "web_search_call" or "action" not in obj:
                new_response_output.append(obj)
                continue
            action = cast("dict[str, Any]", obj["action"])

            # Fix 1: ActionOpenPage with null URL
            if action.get("type") == "open_page" and action.get("url") is None:
                # Convert null to "NULL"
                action["url"] = "NULL"
                self.logger.info("Fixed ActionOpenPage with null URL")

            # Fix 2: Convert find_in_page to find (since OpenAI forgot to add find_in_page)
            elif action.get("type") == "find_in_page":
                action["type"] = "find"  # Convert to supported type
                self.logger.info("Fixed find_in_page to find")

            new_response_output.append(obj)

        return new_response_output

    async def display_batch_result(self, console: Console, batch_result: BatchResponse, index: int) -> None:
        """Display detailed information for a single batch result."""
        custom_id = batch_result.custom_id

        if batch_result.error:
            error_panel = Panel(
                f"[red]Error in batch result:[/red]\n{batch_result.error}",
                title=f"[bold red]Result #{index}: {custom_id}[/bold red]",
                border_style="red",
                expand=True,
            )
            console.print(error_panel)
            console.print("")
            return

        if not batch_result.response:
            console.print(f"[yellow]No response data for result #{index}: {custom_id}[/yellow]")
            console.print("")
            return

        response = batch_result.response.body

        # Create main result panel with metadata
        metadata_info: list[str] = []
        metadata_info.append(f"[bold blue]Response ID:[/bold blue] {response.id}")
        metadata_info.append(f"[bold blue]Model:[/bold blue] {response.model}")
        metadata_info.append(f"[bold blue]Status:[/bold blue] {response.status}")
        metadata_info.append(f"[bold blue]Request ID:[/bold blue] {batch_result.response.request_id}")

        if response.usage:
            usage = response.usage
            metadata_info.append(
                f"[bold blue]Token Usage:[/bold blue] Input: {usage.input_tokens}, Output: {usage.output_tokens}, Total: {usage.total_tokens}"
            )
            if usage.output_tokens_details and usage.output_tokens_details.reasoning_tokens:
                metadata_info.append(
                    f"[bold blue]Reasoning Tokens:[/bold blue] {usage.output_tokens_details.reasoning_tokens}"
                )

        if response.max_output_tokens:
            metadata_info.append(f"[bold blue]Max Output Tokens:[/bold blue] {response.max_output_tokens}")

        metadata_panel = Panel(
            "\n".join(metadata_info),
            title=f"[bold cyan]Result #{index}: {custom_id}[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
        console.print(metadata_panel)

        # Display each output item
        for output_idx, output_item in enumerate(response.output, 1):
            await self.display_output_item(console, output_item, output_idx)

        console.print("")  # Add spacing between results

    async def display_output_item(self, console: Console, output_item: ResponseOutputItem, item_idx: int) -> None:
        """Display a single output item with appropriate formatting."""
        item_type = output_item.type
        item_id = getattr(output_item, "id", "no-id")

        if isinstance(output_item, ResponseReasoningItem):
            await self.display_reasoning_item(console, output_item, item_idx)
        elif isinstance(output_item, ResponseFunctionWebSearch):
            await self.display_web_search_item(console, output_item, item_idx)
        elif isinstance(output_item, ResponseOutputMessage):
            await self.display_message_item(console, output_item, item_idx)
        else:
            # Handle unknown item types
            console.print(
                Panel(
                    "\n".join(
                        [
                            f"[yellow]Unknown output item type: {item_type}[/yellow]",
                            f"ID: {item_id}",
                            f"Raw data: {str(output_item)[:500]}...",
                        ]
                    ),
                    title=f"[bold yellow]Output #{item_idx} ({item_type})[/bold yellow]",
                    border_style="yellow",
                )
            )

    async def display_reasoning_item(self, console: Console, item: ResponseReasoningItem, item_idx: int) -> None:
        """Display reasoning output item."""
        content_parts: list[str] = []
        content_parts.append(f"[bold]ID:[/bold] {item.id}")

        if item.status:
            content_parts.append(f"[bold]Status:[/bold] {item.status}")

        if item.summary:
            content_parts.append(f"[bold]Summary:[/bold] {len(item.summary)} items")
            content_parts.extend(f"  • {summary_item.text}" for summary_item in item.summary)
        else:
            content_parts.append("[dim]No reasoning summary available[/dim]")

        if item.content:
            content_parts.append(f"[bold]Content:[/bold] {len(item.content)} items")
            content_parts.extend(f"  • {content_item.text}" for content_item in item.content)

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold magenta]Output #{item_idx}: Reasoning[/bold magenta]",
            border_style="magenta",
        )
        console.print(panel)

    async def display_web_search_item(self, console: Console, item: ResponseFunctionWebSearch, item_idx: int) -> None:
        """Display web search output item."""
        content_parts: list[str] = []
        content_parts.append(f"[bold]ID:[/bold] {item.id}")
        content_parts.append(f"[bold]Status:[/bold] {item.status}")

        action = item.action
        action_type = action.type
        content_parts.append(f"[bold]Action Type:[/bold] {action_type}")

        match action:
            case ActionSearch(query=query, sources=sources):
                content_parts.append(f"[bold]Query:[/bold] {query}")
                if sources:
                    content_parts.append(f"[bold]Sources:[/bold] {len(sources)} found")
                    content_parts.extend(f"  • {source.url}" for source in sources[:3])  # Show first 3 sources
            case ActionOpenPage(url=url):
                content_parts.append(f"[bold]URL:[/bold] {url}")

            case ActionFind(pattern=pattern, url=url):
                content_parts.append(f"[bold]Pattern:[/bold] {pattern}")
                content_parts.append(f"[bold]URL:[/bold] {url}")

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold green]Output #{item_idx}: Web Search[/bold green]",
            border_style="green",
        )
        console.print(panel)

    async def display_message_item(self, console: Console, item: ResponseOutputMessage, item_idx: int) -> None:
        """Display message output item with content and annotations."""
        content_parts: list[str] = []
        content_parts.append(f"[bold]ID:[/bold] {item.id}")
        content_parts.append(f"[bold]Role:[/bold] {item.role}")
        content_parts.append(f"[bold]Status:[/bold] {item.status}")
        content_parts.append("")

        for content_idx, content_item in enumerate(item.content, 1):
            if isinstance(content_item, ResponseOutputText):
                text = content_item.text

                # Decode Unicode escape sequences
                with contextlib.suppress(UnicodeDecodeError, UnicodeEncodeError):
                    text = text.encode("utf-8").decode("utf-8")

                content_parts.append(f"[bold]Text Content #{content_idx}:[/bold]")
                content_parts.append(text)
                content_parts.append("")

                # Show annotations (citations)
                if content_item.annotations:
                    content_parts.append(f"[bold]Citations ({len(content_item.annotations)}):[/bold]")
                    for ann_idx, annotation in enumerate(
                        content_item.annotations[: deep_research_settings.citation_display_limit], 1
                    ):  # Show first N
                        if annotation.type == "url_citation":
                            content_parts.append(
                                f"  {ann_idx}. [{annotation.start_index}-{annotation.end_index}] {annotation.title}"
                            )
                            content_parts.append(f"     {annotation.url}")
                    if len(content_item.annotations) > deep_research_settings.citation_display_limit:
                        content_parts.append(
                            f"     ... and {len(content_item.annotations) - deep_research_settings.citation_display_limit} more citations"
                        )
            else:
                content_parts.append(f"[yellow]Unknown content type: {content_item.type}[/yellow]")

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold blue]Output #{item_idx}: Assistant Message[/bold blue]",
            border_style="blue",
            expand=True,
        )
        console.print(panel)

    @override
    async def cli_cmd_async(self) -> None:
        """Retrieve and display batch results in a TUI."""
        # Get Batch object
        batch_job = await self.client.batches.retrieve(self.batch_job_id)
        # Download results
        if not batch_job.output_file_id:
            self.logger.error("No output file ID in completed batch job")
            return

        file_content = await self.client.files.content(batch_job.output_file_id)
        results_text = file_content.text

        # Parse JSONL results
        results: list[BatchResponse] = []
        for i, line in enumerate(results_text.strip().splitlines()):
            if line.strip():
                try:
                    full_json_obj = orjson.loads(line)  # pyright: ignore[reportAny]
                    full_json_obj["response"]["body"]["output"] = self.fix_response_data(
                        full_json_obj["response"]["body"]["output"]  # pyright: ignore[reportAny]
                    )
                    result = BatchResponse.model_validate(full_json_obj)
                    results.append(result)
                except ValidationError:
                    self.logger.exception("Failed to parse result line %d", i)

        # Display in TUI
        console = Console()

        if not results:
            console.print(Panel("No results found", title="Batch Results", style="red"))
            return

        console.print(f"\n[bold green]Batch job {batch_job.id} completed with {len(results)} results[/bold green]\n")

        for i, batch_result in enumerate(results, 1):
            await self.display_batch_result(console, batch_result, i)


class Create(CommonCliSettings):
    prompt_files: CliPositionalArg[list[str]]
    system_prompt_path: Path = Path(__file__).parent / "system_prompts" / "market_analysis.md"

    @computed_field
    @property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=deep_research_settings.openai_api_key)

    async def load_system_prompt(self) -> str:
        async with await open_file(self.system_prompt_path, "r", encoding="utf-8") as f:
            return await f.read()

    async def craft_request(self, system_prompt: str, prompt: str) -> ResponseInputParam:
        return [
            Message(content=[ResponseInputTextParam(text=system_prompt, type="input_text")], role="system"),
            Message(content=[ResponseInputTextParam(text=prompt, type="input_text")], role="user"),
        ]

    async def prepare_batch_requests(self, prompts: list[str]) -> list[MyBatchRequest]:
        batch_requests: list[MyBatchRequest] = []
        # Use a simple for-loop and await craft_request for each prompt.
        system_prompt = await self.load_system_prompt()
        for prompt in prompts:
            messages = await self.craft_request(system_prompt, prompt)
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
        return batch_requests

    async def busy_wait_for_batch_job(self, job: Batch, interval_s: float = 10.0) -> Batch:
        progress = create_progress()
        with progress:
            task_id = progress.add_task(f"Waiting for batch job {job.id} to complete", total=None)
            while True:
                batch_job = await self.client.batches.retrieve(job.id)
                progress.update(task_id, description=f"Waiting for batch job {batch_job.id}: {batch_job.status}")
                if batch_job.status == "completed":
                    return batch_job
                await anyio.sleep(interval_s)

    @override
    async def cli_cmd_async(self) -> None:
        prompts: list[str] = []
        for prompt_file in self.prompt_files:
            async with await open_file(Path(prompt_file), "r", encoding="utf-8") as f:
                prompts.append(await f.read())

        self.logger.info("Loaded %d prompts. Preparing the batch request...", len(prompts))
        batch_requests = await self.prepare_batch_requests(prompts)

        # Upload the file to openai
        self.logger.info("Uploading file to openai")
        batch_file = await self.client.files.create(
            file=BytesIO("\n".join(request.model_dump_json() for request in batch_requests).encode("utf-8")),
            purpose="batch",
        )
        batch_job = await self.client.batches.create(
            completion_window="24h",
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
        )
        self.logger.info("Batch job created with ID %s", batch_job.id)
        completed_job = await self.busy_wait_for_batch_job(batch_job)
        self.logger.info("Batch job %s completed", completed_job.id)

        # Display results using Retrieve subcommand logic
        await Retrieve(log_level=self.log_level, batch_job_id=completed_job.id).cli_cmd_async()


class Cli(CommonCliSettings):
    create: CliSubCommand[Create]
    retrieve: CliSubCommand[Retrieve]


if __name__ == "__main__":
    _ = CliApp.run(Cli)
