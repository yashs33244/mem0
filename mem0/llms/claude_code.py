"""LLM provider that runs Claude Code headless (``claude -p``) instead of calling an HTTP API.

It needs no API key: the CLI reuses the login stored by Claude Code. Every
``generate_response`` call is one subprocess invocation, so it is slower than a
direct API call (a few seconds) but works on any machine where ``claude`` is
logged in.
"""

import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.claude_code import ClaudeCodeConfig
from mem0.llms.base import LLMBase

DEFAULT_MODEL = "claude-haiku-4-5"

JSON_OBJECT_INSTRUCTION = (
    "Reply with a single JSON object and nothing else: no prose before or after it, no markdown, no code fences."
)

TOOL_CALL_INSTRUCTION = (
    "Reply with a single JSON object and nothing else, in exactly this shape: "
    '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments matching the schema>}}], '
    '"content": null}. Use "content" for any plain-text answer.'
)

# Environment variables that mark "running inside a Claude Code session". They are removed
# for the child so the CLI does not refuse to start or behave like a nested session.
_STRIP_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", re.DOTALL)


class ClaudeCodeError(RuntimeError):
    """Raised when the Claude Code CLI cannot be run, times out, exits non-zero, or reports an error."""


def _content_text(content) -> str:
    """Flatten a message content field (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` in ``text`` that parses as JSON, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate, strict=False)
                        return candidate
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def extract_json_object(text: str) -> str:
    """Best-effort extraction of one JSON object from model output.

    Handles bare JSON, JSON inside ``` fences, and JSON surrounded by prose.
    Falls back to the fence-stripped text when nothing parses so the caller's own
    parser produces the error.
    """
    stripped = text.strip()
    try:
        json.loads(stripped, strict=False)
        return stripped
    except ValueError:
        pass
    match = _FENCE_RE.search(stripped)
    if match:
        stripped = match.group(1).strip()
        try:
            json.loads(stripped, strict=False)
            return stripped
        except ValueError:
            pass
    found = _first_json_object(stripped)
    return found if found is not None else stripped


def _describe_tools(tools: List[Dict], tool_choice: str) -> str:
    lines = ["You have the following tools available:"]
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = spec.get("name", "")
        description = spec.get("description", "")
        schema = spec.get("parameters") or spec.get("input_schema") or {}
        lines.append(f"- {name}: {description}")
        lines.append(f"  arguments JSON schema: {json.dumps(schema)}")
    if tool_choice in ("required", "any"):
        lines.append("You must call at least one tool.")
    else:
        lines.append('Call a tool only when it applies; otherwise return an empty "tool_calls" list.')
    lines.append(TOOL_CALL_INSTRUCTION)
    return "\n".join(lines)


class ClaudeCodeLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, ClaudeCodeConfig, Dict]] = None):
        if config is None:
            config = ClaudeCodeConfig()
        elif isinstance(config, dict):
            config = ClaudeCodeConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, ClaudeCodeConfig):
            config = ClaudeCodeConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client_proxies,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = DEFAULT_MODEL

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response by running ``claude -p`` once.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
                System messages become the CLI ``--system-prompt``; the remaining turns
                form the prompt (a single user turn is sent as-is, several turns are
                labelled "User:" / "Assistant:").
            response_format (dict, optional): ``{"type": "json_object"}`` makes the
                provider ask for JSON and return only the extracted JSON object text.
            tools (list, optional): Tool definitions (OpenAI or Anthropic shape). When
                given, the tools are described in the prompt and the return value is
                ``{"content": ..., "tool_calls": [{"name", "arguments"}]}``.
            tool_choice (str, optional): "auto", "required"/"any", or "none".
            **kwargs: Ignored (the CLI exposes no sampling parameters).

        Returns:
            str or dict: The model text, the extracted JSON text, or the tool-call dict.
        """
        use_tools = bool(tools) and tool_choice != "none"
        wants_json = isinstance(response_format, dict) and response_format.get("type") in (
            "json_object",
            "json_schema",
        )

        system_prompt, prompt = self._build_prompt(messages)
        if use_tools:
            prompt = f"{prompt}\n\n{_describe_tools(tools, tool_choice)}"
        elif wants_json:
            prompt = f"{prompt}\n\n{JSON_OBJECT_INSTRUCTION}"

        raw = self._run(prompt, system_prompt)

        if use_tools:
            return self._parse_tool_calls(raw)
        if wants_json:
            return extract_json_object(raw)
        return raw

    @staticmethod
    def _build_prompt(messages: List[Dict]):
        system_parts = []
        turns = []
        for message in messages:
            role = message.get("role", "user")
            text = _content_text(message.get("content", ""))
            if role == "system":
                system_parts.append(text)
            else:
                turns.append((role, text))

        system_prompt = "\n\n".join(p for p in system_parts if p)
        if len(turns) == 1 and turns[0][0] == "user":
            prompt = turns[0][1]
        else:
            prompt = "\n\n".join(f"{role.capitalize()}:\n{text}" for role, text in turns)
        return system_prompt, prompt

    def _command(self, system_prompt: str) -> List[str]:
        cmd = [
            self.config.claude_bin,
            "-p",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            self.config.model,
            "--max-turns",
            str(self.config.max_turns),
            *self.config.extra_args,
        ]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        return cmd

    def _run(self, prompt: str, system_prompt: str) -> str:
        env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
        env["MEM0_CLAUDE_HEADLESS"] = "1"
        cmd = self._command(system_prompt)
        try:
            # The prompt goes through stdin: it can be large and must not hit ARG_MAX.
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                timeout=self.config.timeout,
            )
        except FileNotFoundError as e:
            raise ClaudeCodeError(
                f"Claude Code executable not found: {self.config.claude_bin!r}. "
                "Install Claude Code or set claude_bin in the llm config."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(
                f"Claude Code timed out after {self.config.timeout}s (model {self.config.model})"
            ) from e

        envelope = None
        try:
            envelope = json.loads(proc.stdout)
        except ValueError:
            pass
        result = envelope.get("result") if isinstance(envelope, dict) else None

        if proc.returncode != 0 or (isinstance(envelope, dict) and envelope.get("is_error")):
            detail = result or proc.stderr.strip() or proc.stdout.strip() or "no output"
            raise ClaudeCodeError(f"Claude Code exited with status {proc.returncode}: {detail[:2000]}")
        if not isinstance(envelope, dict):
            raise ClaudeCodeError(f"Claude Code returned no JSON envelope: {proc.stdout[:500]!r}")
        if result is None:
            raise ClaudeCodeError(f"Claude Code envelope has no 'result' field: {proc.stdout[:500]!r}")
        return result if isinstance(result, str) else json.dumps(result)

    @staticmethod
    def _parse_tool_calls(raw: str) -> Dict:
        result = {"content": None, "tool_calls": []}
        try:
            data = json.loads(extract_json_object(raw), strict=False)
        except ValueError:
            result["content"] = raw
            return result
        if not isinstance(data, dict):
            result["content"] = raw
            return result

        if isinstance(data.get("content"), str):
            result["content"] = data["content"]
        for call in data.get("tool_calls") or []:
            if not isinstance(call, dict) or not call.get("name"):
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments, strict=False)
                except ValueError:
                    arguments = {"input": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            result["tool_calls"].append({"name": call["name"], "arguments": arguments})

        if not result["tool_calls"] and result["content"] is None:
            result["content"] = raw
        return result
