"""Modular prompt loading system with safe Jinja2 template rendering."""

from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from jinja2 import BaseLoader, Environment, meta, nodes
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)
from pydantic import BaseModel, Field, field_validator

from logs import get_logger

logger = get_logger(__name__)


class PromptMessage(BaseModel):
    """A single message in an agent prompt with role and templated content."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., description="Jinja2 template string")

    @field_validator("content")
    @classmethod
    def validate_template_syntax(cls, v: str) -> str:
        """Validate that the content is a valid Jinja2 template."""
        try:
            env = Environment(loader=BaseLoader(), autoescape=False)  # noqa: S701
            _ = env.parse(v)
        except Exception as e:
            msg = f"Invalid Jinja2 template syntax: {e}"
            raise ValueError(msg) from e
        return v


class AgentPrompt(BaseModel):
    """Complete agent prompt configuration with metadata and messages."""

    name: str = Field(..., description="Agent name")
    messages: list[PromptMessage] = Field(..., description="List of prompt messages")

    @field_validator("messages")
    @classmethod
    def validate_safe_templates(cls, v: list[PromptMessage]) -> list[PromptMessage]:
        """Validate templates for security - only allow safe operations."""
        env = Environment(loader=BaseLoader(), autoescape=False)  # noqa: S701

        # Define allowed node types for safe template evaluation
        safe_nodes = {
            nodes.Template,
            nodes.Output,
            nodes.TemplateData,
            nodes.Name,
            nodes.Getattr,
            nodes.Getitem,
            nodes.Const,
            nodes.List,
            nodes.Tuple,
            nodes.Dict,
            nodes.If,
            nodes.For,
            nodes.Filter,
            nodes.Test,
        }

        for msg in v:
            try:
                ast = env.parse(msg.content)

                # Check all nodes in the AST
                for node in ast.find_all(nodes.Node):
                    if type(node) not in safe_nodes:
                        error_msg = f"Unsafe template operation: {type(node).__name__}"
                        raise ValueError(error_msg)  # noqa: TRY301

                # Check for dangerous template variables/attributes
                variables = meta.find_undeclared_variables(ast)
                dangerous_vars = {"__", "import", "exec", "eval", "open", "file"}

                for var in variables:
                    if any(danger in var.lower() for danger in dangerous_vars):
                        error_msg = f"Potentially unsafe template variable: {var}"
                        raise ValueError(error_msg)  # noqa: TRY301

            except Exception as e:
                validation_msg = f"Template security validation failed for {msg.role} message: {e}"
                raise ValueError(validation_msg) from e

        return v


async def load_component_by_path(prompts_dir: Path, relative_path: str) -> str:
    """Load a component by its relative path (e.g., 'document_making/basic_role' or 'examples/1')."""
    file_path = prompts_dir / f"{relative_path}.md"
    try:
        if file_path.exists():
            async with await anyio.open_file(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return content.strip()
        else:
            logger.warning("Component file not found: %s", file_path)
            return ""
    except Exception:
        logger.exception("Error loading component %s", relative_path)
        return ""


class TemplateContext:
    """Safe context manager for template rendering with component loading."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir: Path = prompts_dir
        self._component_cache: dict[str, str] = {}

    async def get_context(self, **extra_vars: Any) -> dict[str, Any]:  # noqa: ANN401  # pyright: ignore[reportAny]
        """Build safe context for template rendering with nested component structure."""
        context: dict[str, Any] = {}

        # Pre-load all components recursively using nested structure
        for file_path in self.prompts_dir.rglob("*.md"):
            # Skip agent files as they're not components
            if file_path.parent.name == "agents":
                continue

            # Create relative path (e.g., "document_making/basic_role", "examples/1")
            relative_path = file_path.relative_to(self.prompts_dir).with_suffix("").as_posix()

            # Check cache first
            if relative_path not in self._component_cache:
                content = await load_component_by_path(self.prompts_dir, relative_path)
                self._component_cache[relative_path] = content

            # Build nested structure: document_making/basic_role → context["document_making"]["basic_role"]
            path_parts = relative_path.split("/")
            current_level = context

            for part in path_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Set the final content
            current_level[path_parts[-1]] = self._component_cache[relative_path]

        # Add any extra variables passed in
        context.update(extra_vars)

        return context


async def load_agent_prompt(agent_file: Path) -> AgentPrompt:
    """Load and parse an agent prompt from a YAML file."""
    try:
        async with await anyio.open_file(agent_file, "r", encoding="utf-8") as f:
            content = await f.read()

        data: list[dict[str, str]] = yaml.safe_load(content)  # pyright: ignore[reportAny]
        return AgentPrompt.model_validate({"name": agent_file.stem, "messages": data})

    except Exception:
        logger.exception("Error loading agent from %s", agent_file)
        raise


async def render_agent_prompt(
    agent: AgentPrompt,
    context: TemplateContext,
    **template_vars: Any,  # noqa: ANN401  # pyright: ignore[reportAny]
) -> list[AllMessageValues]:
    """Safely render agent prompt with component substitution."""
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,  # We want raw text output for markdown  # noqa: S701
        enable_async=True,
    )

    # Build render context
    render_context = await context.get_context(**template_vars)

    rendered_messages: list[AllMessageValues] = []
    for msg in agent.messages:
        try:
            template = env.from_string(msg.content)
            rendered_content = await template.render_async(render_context)
            content = rendered_content.strip()
            if msg.role == "system":
                sys_msg: ChatCompletionSystemMessage = {"role": "system", "content": content}
                rendered_messages.append(sys_msg)
            elif msg.role == "assistant":
                asst_msg: ChatCompletionAssistantMessage = {"role": "assistant", "content": content}
                rendered_messages.append(asst_msg)
            else:
                user_msg: ChatCompletionUserMessage = {"role": "user", "content": content}
                rendered_messages.append(user_msg)
        except Exception:
            logger.exception("Error rendering template for %s message", msg.role)
            # Fall back to original content if rendering fails
            fallback_content = msg.content
            if msg.role == "system":
                sys_msg_fb: ChatCompletionSystemMessage = {"role": "system", "content": fallback_content}
                rendered_messages.append(sys_msg_fb)
            elif msg.role == "assistant":
                asst_msg_fb: ChatCompletionAssistantMessage = {"role": "assistant", "content": fallback_content}
                rendered_messages.append(asst_msg_fb)
            else:
                user_msg_fb: ChatCompletionUserMessage = {"role": "user", "content": fallback_content}
                rendered_messages.append(user_msg_fb)

    return rendered_messages


async def get_rendered_agent(agent_name: str, prompts_dir: Path, **template_vars: Any) -> list[AllMessageValues]:  # noqa: ANN401  # pyright: ignore[reportAny]
    """Load and render an agent prompt by name."""
    agent_file = prompts_dir / "agents" / f"{agent_name}.yaml"

    if not agent_file.exists():
        error_msg = f"Agent file not found: {agent_file}"
        raise FileNotFoundError(error_msg)

    agent = await load_agent_prompt(agent_file)
    context = TemplateContext(prompts_dir)

    return await render_agent_prompt(agent, context, **template_vars)
