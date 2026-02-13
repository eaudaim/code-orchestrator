import json
import sys
import types
import unittest

if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(ResponseError=Exception)

if "rich" not in sys.modules:
    rich_module = types.ModuleType("rich")
    rich_console_module = types.ModuleType("rich.console")

    class _Console:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            pass

    rich_console_module.Console = _Console
    rich_module.console = rich_console_module
    sys.modules["rich"] = rich_module
    sys.modules["rich.console"] = rich_console_module

from core.model import normalize_tool_calls


class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, call_id: str, call_type: str, function: _MockFunction):
        self.id = call_id
        self.type = call_type
        self.function = function


class NormalizeToolCallsTestCase(unittest.TestCase):
    def test_preserves_dict_tool_call(self) -> None:
        raw_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": {"query": "hello"}},
            }
        ]

        normalized_calls = normalize_tool_calls(raw_calls)

        self.assertEqual(normalized_calls, raw_calls)
        json.dumps(normalized_calls)

    def test_converts_tool_call_object_to_serializable_dict(self) -> None:
        raw_calls = [
            _MockToolCall(
                call_id="call_2",
                call_type="function",
                function=_MockFunction(name="search", arguments='{"query": "world"}'),
            )
        ]

        normalized_calls = normalize_tool_calls(raw_calls)

        self.assertEqual(
            normalized_calls,
            [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"query": "world"}},
                }
            ],
        )
        json.dumps(normalized_calls)


if __name__ == "__main__":
    unittest.main()
