import sys
import types
import unittest
from unittest.mock import patch

if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.SimpleNamespace(ResponseError=Exception)

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *args, **kwargs):  # pragma: no cover - test stub
        return iterable if iterable is not None else []

    tqdm_module.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_module

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

import runtime.loop as loop


class _FakeConsole:
    def print(self, *args, **kwargs):
        pass


class AutonomyLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        loop.console = _FakeConsole()
        loop.log_verbose = lambda message: None

    def test_non_autonomy_executes_successive_tool_calls(self) -> None:
        messages = [{"role": "user", "content": "Go"}]
        files_data = {}

        responses = [
            (
                {"role": "assistant", "content": "first"},
                True,
                [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"directory_path": "."},
                        }
                    }
                ],
            ),
            (
                {"role": "assistant", "content": "second"},
                True,
                [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"directory_path": "."},
                        }
                    }
                ],
            ),
            ({"role": "assistant", "content": "done"}, False, []),
        ]

        with patch.object(loop, "AUTONOMY", False), patch.object(
            loop, "MAX_AUTONOMY_ITERATIONS", 20
        ), patch.object(loop, "call_model_and_stream", side_effect=responses) as model_mock, patch.object(
            loop, "list_files_tool", return_value="ok"
        ) as list_files_mock:
            loop.autonomy_loop(messages, files_data)

        self.assertEqual(model_mock.call_count, 3)
        self.assertEqual(list_files_mock.call_count, 2)

    def test_autonomy_stops_when_iteration_limit_is_reached(self) -> None:
        messages = [{"role": "user", "content": "Go"}]
        files_data = {}

        responses = [
            (
                {"role": "assistant", "content": "first"},
                True,
                [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"directory_path": "."},
                        }
                    }
                ],
            ),
            (
                {"role": "assistant", "content": "second"},
                True,
                [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"directory_path": "."},
                        }
                    }
                ],
            ),
        ]

        with patch.object(loop, "AUTONOMY", True), patch.object(
            loop, "MAX_AUTONOMY_ITERATIONS", 2
        ), patch.object(loop, "AUTONOMY_TIMEOUT", 0), patch.object(
            loop, "wait_for_manual_override", return_value=False
        ), patch.object(loop, "call_model_and_stream", side_effect=responses) as model_mock, patch.object(
            loop, "list_files_tool", return_value="ok"
        ) as list_files_mock:
            loop.autonomy_loop(messages, files_data)

        self.assertEqual(list_files_mock.call_count, 2)
        self.assertEqual(model_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
