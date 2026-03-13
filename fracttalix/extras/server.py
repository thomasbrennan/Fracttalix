# fracttalix/extras/server.py
# SentinelServer — asyncio HTTP server wrapping MultiStreamSentinel.

import asyncio
import json
import time
from typing import Any, Optional, Tuple

from fracttalix.config import SentinelConfig
from fracttalix.multistream import MultiStreamSentinel

try:
    from fracttalix import __version__
except Exception:
    __version__ = "12.2.0"


class SentinelServer:
    """Async HTTP server wrapping MultiStreamSentinel.

    Endpoints:
      POST /update/<stream_id>       body: {"value": ...}
      GET  /streams                  returns list of stream IDs
      GET  /status/<stream_id>       returns stream status
      DELETE /stream/<stream_id>     delete a stream
      POST /reset/<stream_id>        reset a stream
      GET  /health                   version + uptime
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 config: Optional[SentinelConfig] = None):
        self.host = host
        self.port = port
        self.mss = MultiStreamSentinel(config=config)
        self._start_time = time.time()
        self._server: Optional[Any] = None

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            lines = raw.decode("utf-8", errors="replace").split("\r\n")
            if not lines:
                writer.close()
                return
            req_line = lines[0].split()
            if len(req_line) < 2:
                writer.close()
                return
            method = req_line[0].upper()
            path = req_line[1]
            # Find body (after blank line)
            body_str = ""
            try:
                sep = raw.index(b"\r\n\r\n")
                body_str = raw[sep + 4:].decode("utf-8", errors="replace").strip()
            except ValueError:
                pass

            status, resp = await self._route(method, path, body_str)
            resp_bytes = json.dumps(resp).encode()
            http = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(resp_bytes)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode() + resp_bytes
            writer.write(http)
            await writer.drain()
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode()
            http = (f"HTTP/1.1 500 Internal Server Error\r\nContent-Length: {len(err)}\r\n\r\n").encode() + err
            writer.write(http)
            await writer.drain()
        finally:
            writer.close()

    async def _route(self, method: str, path: str,
                     body: str) -> Tuple[str, Any]:
        parts = [p for p in path.strip("/").split("/") if p]

        if method == "GET" and parts == ["health"]:
            try:
                from fracttalix import __version__ as _ver
            except Exception:
                _ver = "12.2.0"
            return "200 OK", {
                "version": _ver,
                "uptime": time.time() - self._start_time,
                "streams": len(self.mss.list_streams()),
            }

        if method == "GET" and parts == ["streams"]:
            return "200 OK", {"streams": self.mss.list_streams()}

        if method == "GET" and len(parts) == 2 and parts[0] == "status":
            return "200 OK", self.mss.status(parts[1])

        if method == "POST" and len(parts) == 2 and parts[0] == "update":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return "400 Bad Request", {"error": "invalid JSON"}
            value = payload.get("value")
            if value is None:
                return "400 Bad Request", {"error": "missing 'value'"}
            result = await self.mss.aupdate(parts[1], value)
            return "200 OK", result

        if method == "DELETE" and len(parts) == 2 and parts[0] == "stream":
            ok = self.mss.delete_stream(parts[1])
            return ("200 OK", {"deleted": parts[1]}) if ok else ("404 Not Found", {"error": "not found"})

        if method == "POST" and len(parts) == 2 and parts[0] == "reset":
            ok = self.mss.reset_stream(parts[1])
            return ("200 OK", {"reset": parts[1]}) if ok else ("404 Not Found", {"error": "not found"})

        return "404 Not Found", {"error": f"no route for {method} {path}"}

    async def serve_forever(self) -> None:
        """Start async server (runs until cancelled)."""
        self._server = await asyncio.start_server(
            self._handle, self.host, self.port
        )
        async with self._server:
            await self._server.serve_forever()

    def run(self) -> None:
        """Blocking entry point (wraps asyncio.run)."""
        try:
            asyncio.run(self.serve_forever())
        except KeyboardInterrupt:
            pass
