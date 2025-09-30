from pathlib import Path
from typing import Any, override

from litellm.router import Router
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LLMDeployment(BaseModel):
    """Represents a single LLM provider/model configuration.

    - model: LiteLLM model identifier (e.g., "gemini/gemini-2.5-pro" or "openrouter/google/gemini-2.5-pro")
    - is_paid: whether this deployment may incur cost
    """

    model: str
    is_paid: bool = False


class Settings(BaseSettings):
    # Back-compat single-model settings (used to derive defaults for deployments)
    # drafting_model: str = "gemini/gemini-2.5-pro"
    drafting_model: str = "openrouter/google/gemini-2.5-pro"
    fixing_model: str = "openrouter/x-ai/grok-4-fast:free"

    # New deployment lists (ordered preference)
    drafting_deployments: list[LLMDeployment] | None = None
    fixing_deployments: list[LLMDeployment] | None = None

    # Control whether paid providers are allowed as fallback when free ones fail
    allow_paid_fallback: bool = False

    max_concurrency: int = 16
    request_timeout_s: float = 600.0
    output_extension: str = "md"
    max_retry_attempts: int = 5

    enable_fixing_phase: bool = True
    # Context configuration for fixing
    context_lines: int = 5  # Number of context lines before/after Typst blocks for LLM fixing

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny]
        self.system_prompt_path = self.system_prompt_path.resolve().absolute()

    system_prompt_path: Path = Path(__file__).parent / "system_prompt.md"

    def _derive_default_deployments(self, kind: str) -> list[LLMDeployment]:
        """Return a single-entry deployment list from legacy fields.

        Uses a simple, explicit heuristic: models tagged with ":free" are treated as free.
        """
        model = self.drafting_model if kind == "drafting" else self.fixing_model
        return [LLMDeployment(model=model, is_paid=(":free" not in model))]

    def get_ordered_deployments(self, kind: str) -> list[LLMDeployment]:
        """Return ordered deployments for the given kind ("drafting" | "fixing")."""
        if kind == "drafting":
            return self.drafting_deployments or self._derive_default_deployments(kind)
        if kind == "fixing":
            return self.fixing_deployments or self._derive_default_deployments(kind)
        raise ValueError(f"Unknown deployment kind: {kind}")

    def get_effective_deployments(self, kind: str) -> list[LLMDeployment]:
        """Apply allow_paid_fallback to filter deployments; if that removes all, fall back to full list."""
        deployments = self.get_ordered_deployments(kind)
        if self.allow_paid_fallback:
            return deployments
        filtered = [d for d in deployments if not d.is_paid]
        return filtered if filtered else deployments

    def build_router_and_preferred(self, kind: str) -> tuple[Router, str]:
        """Construct a LiteLLM Router from effective deployments and return (router, preferred_model).

        Returned router is typed as object to avoid tight coupling to litellm types at import time.
        """
        deployments = self.get_effective_deployments(kind)
        if not deployments:
            # Should never happen due to derivation fallback
            raise RuntimeError(f"No deployments configured for kind={kind}")

        model_list = [{"model_name": d.model, "litellm_params": {"model": d.model}} for d in deployments]
        router = Router(model_list=model_list)
        preferred_model = deployments[0].model
        return router, preferred_model


settings = Settings()
