from typing import List, Optional

from mem0.configs.llms.base import BaseLlmConfig


class ClaudeCodeConfig(BaseLlmConfig):
    """
    Configuration class for the Claude Code headless provider.

    The provider shells out to the ``claude`` CLI (``claude -p``) and reuses the
    login already stored by Claude Code, so no API key is required. Sampling
    parameters such as temperature and max_tokens are accepted for
    compatibility with the other providers but are not sent to the CLI.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # Claude Code specific parameters
        claude_bin: str = "claude",
        timeout: int = 120,
        max_turns: int = 1,
        extra_args: Optional[List[str]] = None,
    ):
        """
        Initialize Claude Code configuration.

        Args:
            model: Model passed to ``claude --model``, defaults to "claude-haiku-4-5"
            temperature: Accepted for compatibility, not used by the CLI
            api_key: Accepted for compatibility, not used by the CLI
            max_tokens: Accepted for compatibility, not used by the CLI
            top_p: Accepted for compatibility, not used by the CLI
            top_k: Accepted for compatibility, not used by the CLI
            enable_vision: Accepted for compatibility, not used by the CLI
            vision_details: Accepted for compatibility, not used by the CLI
            http_client_proxies: Accepted for compatibility, not used by the CLI
            claude_bin: Path or name of the Claude Code executable, defaults to "claude"
            timeout: Seconds to wait for one CLI call before failing, defaults to 120
            max_turns: Value passed to ``claude --max-turns``, defaults to 1
            extra_args: Additional CLI flags. Defaults to ["--safe-mode"], which
                disables hooks, plugins and MCP servers while keeping the stored
                login. Use ["--bare"] instead when ANTHROPIC_API_KEY is set.
        """
        super().__init__(
            model=model or "claude-haiku-4-5",
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )

        self.claude_bin = claude_bin
        self.timeout = timeout
        self.max_turns = max_turns
        self.extra_args = list(extra_args) if extra_args is not None else ["--safe-mode"]
