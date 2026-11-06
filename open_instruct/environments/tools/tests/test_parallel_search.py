"""Synthetic MCP regressions; live provider validation is separate."""

import asyncio
import dataclasses
import json
import unittest
from builtins import ExceptionGroup
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
from aiohttp import web
from mcp import McpError
from mcp.types import ErrorData
from transformers import HfArgumentParser

from open_instruct.environments.base import EnvCall
from open_instruct.environments.tools import parallel_search
from open_instruct.environments.tools.tools import TOOL_REGISTRY, SerperSearchTool, YouSearchTool
from open_instruct.environments.tools.utils import EnvsConfig


def registered_tool():
    # Exercise the CLI dataclass and JSON config loader, as used by the trainers.
    parser = HfArgumentParser(cast(Any, EnvsConfig))
    (config,) = parser.parse_args_into_dataclasses(
        ["--tools", "parallel_search", "--tool_call_names", "search", "--tool_configs", '{"timeout": 3}']
    )
    parsed = cast(EnvsConfig, config)._parsed_tools[0]
    typed = TOOL_REGISTRY[parsed.name](**parsed.config)
    kwargs = dataclasses.asdict(typed)
    kwargs.pop("pool_size")  # Consumed by EnvironmentPool, not the tool constructor.
    return cast(type[parallel_search.ParallelSearchTool], typed.tool_class)(call_name=parsed.call_name, **kwargs)


class TestParallelSearch(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tool = registered_tool()
        self.call = EnvCall(id="test", name="search", args={"query": "Python asyncio task cancellation documentation"})

    async def test_opt_in_and_credentials(self):
        self.assertFalse(EnvsConfig().enabled)
        self.assertEqual(self.tool.call_name, "search")
        self.assertEqual(self.tool.timeout, 3)
        self.assertEqual(self.tool.get_tool_definitions()[0]["function"]["name"], "search")
        with patch.dict("os.environ", {}, clear=True):
            for tool_class, key in [(SerperSearchTool, "SERPER_API_KEY"), (YouSearchTool, "YDC_API_KEY")]:
                with self.assertRaisesRegex(ValueError, key):
                    tool_class(call_name="search")
        with patch.dict("os.environ", {"YDC_API_KEY": "bad-key", "SERPER_API_KEY": "bad-key"}):
            self.assertEqual(YouSearchTool(call_name="search").api_key, "bad-key")
            self.assertEqual(SerperSearchTool(call_name="search").api_key, "bad-key")
        with self.assertRaisesRegex(ValueError, "positive"):
            parallel_search.ParallelSearchTool("search", timeout=0)

    async def test_empty_query_no_network(self):
        with patch.object(self.tool, "_search", new_callable=AsyncMock) as search:
            result = await self.tool.step(EnvCall(id="empty", name="search", args={"query": "  "}))
            self.assertIn("Empty query", result.metadata["error"])
            search.assert_not_called()

    async def test_error_details_and_timeout(self):
        cases = [
            (McpError(ErrorData(code=-32042, message="Rate limited", data={"retry_after": 30})), "-32042", False),
            (ExceptionGroup("SDK", [ValueError("Malformed MCP response")]), "Malformed MCP response", False),
            (TimeoutError("deadline"), "deadline", True),
            (httpx.ReadTimeout("socket deadline"), "socket deadline", True),
            (McpError(ErrorData(code=408, message="SDK request timeout")), "SDK request timeout", True),
        ]
        for error, message, timed_out in cases:
            with self.subTest(message=message), patch.object(self.tool, "_search", side_effect=error):
                result = await self.tool.step(self.call)
                self.assertIn(message, result.metadata["error"])
                self.assertEqual(result.metadata["timeout"], timed_out)
        with patch.object(self.tool, "_search", return_value=("permission denied", True)):
            result = await self.tool.step(self.call)
            self.assertEqual(result.metadata["error"], "permission denied")
        with patch.object(self.tool, "_search", return_value=("", False)):
            result = await self.tool.step(self.call)
            self.assertEqual(result.metadata["error"], "Query returned no results.")

    async def test_total_deadline_and_cancellation(self):
        async def slow(query):
            await asyncio.sleep(5)

        self.tool.timeout = 0.05
        with patch.object(self.tool, "_search", side_effect=slow):
            result = await self.tool.step(self.call)
            self.assertTrue(result.metadata["timeout"])
        with (
            patch.object(self.tool, "_search", side_effect=asyncio.CancelledError),
            self.assertRaises(asyncio.CancelledError),
        ):
            await self.tool.step(self.call)

    async def test_sdk_http_headers_errors_and_cleanup(self):
        """Real local HTTP with synthetic replies and failing remote teardown."""
        requests = []
        clients = []
        mode = "success"
        passage = "Passage with Unicode: \u0085 \u2028 \u2029."

        async def handler(request):
            payload = await request.json() if request.method == "POST" else {}
            method = payload.get("method")
            requests.append((request.method, method, dict(request.headers), payload))
            if request.method == "DELETE":
                return web.Response(status=500)
            if request.method == "GET":
                return web.Response(status=405)
            if "id" not in payload:
                return web.Response(status=202)
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "serverInfo": {"name": "synthetic", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": []}
            elif mode == "rpc_error":
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "error": {"code": -32042, "message": "Rate limited", "data": {"retry_after": 30}},
                    }
                )
            elif mode == "http_error":
                return web.Response(status=403, text="Forbidden")
            else:
                result = {
                    "content": [{"type": "text", "text": f"{passage}\nSource: https://example.org/source"}],
                    "isError": mode == "tool_error",
                }
            return web.json_response(
                {"jsonrpc": "2.0", "id": payload["id"], "result": result}, headers={"Mcp-Session-Id": "synthetic"}
            )

        app = web.Application()
        app.router.add_route("*", "/mcp", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        original = httpx.AsyncClient.__aenter__

        async def enter(client):
            clients.append(client)
            return await original(client)

        try:
            with (
                patch.object(parallel_search, "PARALLEL_MCP_URL", f"http://127.0.0.1:{port}/mcp"),
                patch.object(httpx.AsyncClient, "__aenter__", enter),
                patch.dict("os.environ", {"PARALLEL_API_KEY": "must-not-send", "YDC_API_KEY": "must-not-send"}),
            ):
                for mode in ["success", "tool_error", "rpc_error", "http_error"]:
                    with self.subTest(mode=mode):
                        result = await self.tool.step(self.call)
                        if mode == "success":
                            self.assertFalse(result.metadata["error"])
                            self.assertIn(passage, result.result)
                            self.assertIn("https://example.org/source", result.result)
                        elif mode == "tool_error":
                            self.assertIn(passage, result.metadata["error"])
                        elif mode == "rpc_error":
                            self.assertIn("-32042", result.metadata["error"])
                            self.assertIn("retry_after", result.metadata["error"])
                        else:
                            self.assertIn("403", result.metadata["error"])
                        self.assertFalse(result.metadata["timeout"])
                        self.assertTrue(all(client.is_closed for client in clients))
            self.assertTrue(any(method == "DELETE" for method, *_ in requests))
            self.assertTrue(any(rpc == "notifications/initialized" for _, rpc, *_ in requests))
            for _method, rpc, headers, payload in requests:
                self.assertEqual(headers["User-Agent"], parallel_search.PARALLEL_USER_AGENT)
                self.assertNotIn("Authorization", headers)
                self.assertNotIn("must-not-send", json.dumps(headers))
                if rpc == "tools/call":
                    self.assertEqual(payload["params"]["name"], "web_search")
                    self.assertEqual(
                        payload["params"]["arguments"],
                        {"objective": self.call.args["query"], "search_queries": [self.call.args["query"]]},
                    )
        finally:
            await runner.cleanup()
