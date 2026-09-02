import json
import subprocess
from unittest.mock import patch

import pytest

from mem0.configs.llms.claude_code import ClaudeCodeConfig
from mem0.llms.claude_code import ClaudeCodeError, ClaudeCodeLLM, extract_json_object


def _envelope(result, is_error=False):
    return json.dumps({"type": "result", "subtype": "success", "is_error": is_error, "result": result})


def _completed(stdout, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def mock_run():
    with patch("mem0.llms.claude_code.subprocess.run") as run:
        yield run


def test_plain_text_response_and_command_shape(mock_run, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    mock_run.return_value = _completed(_envelope("pong"))

    llm = ClaudeCodeLLM(ClaudeCodeConfig(model="claude-haiku-4-5", claude_bin="/opt/claude", timeout=30, max_turns=2))
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "ping"},
    ]

    assert llm.generate_response(messages) == "pong"

    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "/opt/claude"
    assert cmd[1] == "-p"
    assert "--no-session-persistence" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-4-5"
    assert cmd[cmd.index("--max-turns") + 1] == "2"
    assert "--safe-mode" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "You are terse."
    assert kwargs["input"] == "ping"
    assert kwargs["timeout"] == 30
    assert "CLAUDECODE" not in kwargs["env"]
    assert "CLAUDE_CODE_ENTRYPOINT" not in kwargs["env"]
    assert kwargs["env"]["MEM0_CLAUDE_HEADLESS"] == "1"


def test_default_config_values():
    llm = ClaudeCodeLLM({"model": None})
    assert llm.config.model == "claude-haiku-4-5"
    assert llm.config.claude_bin == "claude"
    assert llm.config.timeout == 120
    assert llm.config.max_turns == 1
    assert llm.config.extra_args == ["--safe-mode"]

    llm = ClaudeCodeLLM({"extra_args": ["--bare"]})
    assert llm.config.extra_args == ["--bare"]
    assert "--safe-mode" not in llm._command("")


def test_multi_turn_messages_get_role_labels(mock_run):
    mock_run.return_value = _completed(_envelope("ok"))
    llm = ClaudeCodeLLM()
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": [{"type": "text", "text": "third"}]},
    ]

    llm.generate_response(messages)

    prompt = mock_run.call_args.kwargs["input"]
    assert prompt == "User:\nfirst\n\nAssistant:\nsecond\n\nUser:\nthird"
    assert "--system-prompt" not in mock_run.call_args.args[0]


def test_json_object_with_fenced_output(mock_run):
    fenced = 'Here you go:\n```json\n{"memory": [{"text": "Yash uses Neovim", "event": "ADD"}]}\n```\nDone.'
    mock_run.return_value = _completed(_envelope(fenced))
    llm = ClaudeCodeLLM()

    response = llm.generate_response([{"role": "user", "content": "extract"}], response_format={"type": "json_object"})

    assert json.loads(response) == {"memory": [{"text": "Yash uses Neovim", "event": "ADD"}]}
    assert "single JSON object" in mock_run.call_args.kwargs["input"]


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == '{"a": 1}'
    assert extract_json_object('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_object('Sure. {"a": {"b": "}"}} trailing') == '{"a": {"b": "}"}}'
    assert extract_json_object("no json here") == "no json here"


def test_tools_call_parsing(mock_run):
    tool_json = json.dumps(
        {"tool_calls": [{"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}]}}], "content": None}
    )
    mock_run.return_value = _completed(_envelope(f"```json\n{tool_json}\n```"))
    llm = ClaudeCodeLLM()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities",
                "parameters": {"type": "object", "properties": {"entities": {"type": "array"}}},
            },
        }
    ]

    response = llm.generate_response([{"role": "user", "content": "Alice works at UCSD"}], tools=tools)

    assert response == {
        "content": None,
        "tool_calls": [{"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}]}}],
    }
    prompt = mock_run.call_args.kwargs["input"]
    assert "extract_entities" in prompt
    assert "Extract entities" in prompt
    assert '"tool_calls"' in prompt


def test_tools_without_tool_calls_keeps_text_content(mock_run):
    mock_run.return_value = _completed(_envelope("I could not find any entities."))
    llm = ClaudeCodeLLM()
    tools = [{"name": "extract_entities", "description": "Extract entities", "input_schema": {"type": "object"}}]

    response = llm.generate_response([{"role": "user", "content": "nothing here"}], tools=tools)

    assert response == {"content": "I could not find any entities.", "tool_calls": []}


def test_non_zero_exit_raises(mock_run):
    mock_run.return_value = _completed("", returncode=1, stderr="boom: something broke")
    llm = ClaudeCodeLLM()

    with pytest.raises(ClaudeCodeError, match="status 1: boom: something broke"):
        llm.generate_response([{"role": "user", "content": "hi"}])


def test_error_envelope_raises_with_cli_message(mock_run):
    mock_run.return_value = _completed(_envelope("Not logged in. Please run /login", is_error=True), returncode=1)
    llm = ClaudeCodeLLM()

    with pytest.raises(ClaudeCodeError, match="Not logged in"):
        llm.generate_response([{"role": "user", "content": "hi"}])


def test_timeout_raises(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=5)
    llm = ClaudeCodeLLM(ClaudeCodeConfig(timeout=5))

    with pytest.raises(ClaudeCodeError, match="timed out after 5s"):
        llm.generate_response([{"role": "user", "content": "hi"}])


def test_missing_binary_raises(mock_run):
    mock_run.side_effect = FileNotFoundError()
    llm = ClaudeCodeLLM(ClaudeCodeConfig(claude_bin="/nonexistent/claude"))

    with pytest.raises(ClaudeCodeError, match="not found"):
        llm.generate_response([{"role": "user", "content": "hi"}])


def test_factory_registration():
    from mem0.llms.configs import LlmConfig
    from mem0.utils.factory import LlmFactory

    assert "claude_code" in LlmFactory.get_supported_providers()
    llm = LlmFactory.create("claude_code", {"model": "claude-haiku-4-5", "timeout": 7})
    assert isinstance(llm, ClaudeCodeLLM)
    assert llm.config.timeout == 7
    assert LlmConfig(provider="claude_code", config={"model": "claude-haiku-4-5"}).provider == "claude_code"
