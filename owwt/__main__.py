"""Start the web UI on HTTP, and on HTTPS when a certificate exists (browsers only allow the microphone over HTTPS)."""
import asyncio
import contextlib
import signal

import uvicorn

from .config import CERT_DIR, HTTP_PORT, HTTPS_PORT


class Server(uvicorn.Server):
    """Two servers share one event loop, so signals are handled once below instead of by each server."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield

    def install_signal_handlers(self):
        pass


async def serve():
    from .web import app, jobs

    configs = [uvicorn.Config(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")]
    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    if cert.exists() and key.exists():
        configs.append(uvicorn.Config(app, host="0.0.0.0", port=HTTPS_PORT, log_level="warning",
                                      ssl_certfile=str(cert), ssl_keyfile=str(key)))
    servers = [Server(c) for c in configs]

    def stop(*_):
        for server in servers:
            server.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:
            signal.signal(sig, stop)

    print(f"openWakeWord Trainer listening on http://0.0.0.0:{HTTP_PORT}", flush=True)
    if len(servers) > 1:
        print(f"HTTPS (needed for the microphone) on https://0.0.0.0:{HTTPS_PORT}", flush=True)
    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        jobs.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
