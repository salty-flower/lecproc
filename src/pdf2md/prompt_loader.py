"""Modular prompt loading system with safe Jinja2 template rendering."""

import asyncio
import re
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from jinja2 import BaseLoader, Environment, meta, nodes
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


class ComponentLoader:
    """Lazy loader for prompt components in a specific category."""

    def __init__(self, category: str, template_ctx: "TemplateContext") -> None:
        self.category: str = category
        self.template_ctx: TemplateContext = template_ctx
        self._cache: dict[str, str] = {}

    def __getattr__(self, name: str) -> str:
        """Load component on attribute access (e.g., document_making.tikz_diagram)."""
        if name.startswith("_"):
            error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(error_msg)

        cache_key = f"{self.category}.{name}"
        if cache_key not in self._cache:
            # Use asyncio.run to handle async loading in sync context
            self._cache[cache_key] = asyncio.run(self.template_ctx.load_component(self.category, name))

        return self._cache[cache_key]


class ExampleLoader:
    """Lazy loader for numbered examples."""

    def __init__(self, template_ctx: "TemplateContext") -> None:
        self.template_ctx: TemplateContext = template_ctx
        self._cache: dict[int, str] = {}

    def __getitem__(self, index: int) -> str:
        """Load example by index (e.g., examples[1])."""
        if index not in self._cache:
            self._cache[index] = asyncio.run(self.template_ctx.load_example(index))

        return self._cache[index]


class TemplateContext:
    """Safe context manager for template rendering with component loading."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir: Path = prompts_dir
        self._component_cache: dict[str, str] = {}
        self._example_cache: dict[int, str] = {}

    async def load_component(self, category: str, name: str) -> str:
        """Load a component file from category/name.md."""
        cache_key = f"{category}.{name}"
        if cache_key not in self._component_cache:
            file_path = self.prompts_dir / category / f"{name}.md"
            try:
                if file_path.exists():
                    async with await anyio.open_file(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        self._component_cache[cache_key] = content.strip()
                        logger.debug("Loaded component: %s", cache_key)
                else:
                    logger.warning("Component file not found: %s", file_path)
                    self._component_cache[cache_key] = ""
            except Exception:
                logger.exception("Error loading component %s", cache_key)
                self._component_cache[cache_key] = ""

        return self._component_cache[cache_key]

    async def load_example(self, index: int) -> str:
        """Load an example by index from examples/N.md."""
        if index not in self._example_cache:
            examples_dir = self.prompts_dir / "examples"
            file_path = examples_dir / f"{index}.md"
            try:
                if file_path.exists():
                    async with await anyio.open_file(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        self._example_cache[index] = content.strip()
                        logger.debug("Loaded example: %s", index)
                else:
                    logger.debug("Example file not found: %s", file_path)
                    self._example_cache[index] = ""
            except Exception:
                logger.exception("Error loading example %s", index)
                self._example_cache[index] = ""

        return self._example_cache[index]

    async def get_context(self, **extra_vars: Any) -> dict[str, Any]:  # noqa: ANN401  # pyright: ignore[reportAny]
        """Build safe context for template rendering."""
        context = {
            "document_making": ComponentLoader("document_making", self),
            "typst": ComponentLoader("typst", self),
            "examples": ExampleLoader(self),
        }

        # Add any extra variables passed in
        context.update(extra_vars)

        return context


def discover_examples(prompts_dir: Path) -> dict[int, Path]:
    """Discover numbered example files and return mapping of index to path."""
    examples_dir = prompts_dir / "examples"
    if not examples_dir.exists():
        return {}

    examples: dict[int, Path] = {}
    for file_path in examples_dir.glob("*.md"):
        match = re.match(r"^(\d+)\.md$", file_path.name)
        if match:
            index = int(match.group(1))
            examples[index] = file_path

    return dict(sorted(examples.items()))


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
) -> list[dict[str, str]]:
    """Safely render agent prompt with component substitution."""
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,  # We want raw text output for markdown  # noqa: S701
        enable_async=True,
    )

    # Build render context
    render_context = await context.get_context(**template_vars)

    rendered_messages: list[dict[str, str]] = []
    for msg in agent.messages:
        try:
            template = env.from_string(msg.content)
            rendered_content = await template.render_async(render_context)
            rendered_messages.append({"role": msg.role, "content": rendered_content.strip()})
        except Exception:
            logger.exception("Error rendering template for %s message", msg.role)
            # Fall back to original content if rendering fails
            rendered_messages.append({"role": msg.role, "content": msg.content})

    return rendered_messages


async def get_rendered_agent(agent_name: str, prompts_dir: Path, **template_vars: Any) -> list[dict[str, str]]:  # noqa: ANN401  # pyright: ignore[reportAny]
    """Load and render an agent prompt by name."""
    agent_file = prompts_dir / "agents" / f"{agent_name}.yaml"

    if not agent_file.exists():
        error_msg = f"Agent file not found: {agent_file}"
        raise FileNotFoundError(error_msg)

    agent = await load_agent_prompt(agent_file)
    context = TemplateContext(prompts_dir)

    return await render_agent_prompt(agent, context, **template_vars)
