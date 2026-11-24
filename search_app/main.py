"""FastAPI entry point for the local browser interface."""

import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import SearchService

STATIC = Path(__file__).parent / "static"


class IndexRequest(BaseModel):
    folder: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: str = "hybrid"
    extension: str = ""
    folder: str = ""
    limit: int = Field(default=10, ge=1, le=100)


def create_app(service: SearchService | None = None) -> FastAPI:
    """Allow an isolated index to be provided for automated API tests."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.search = service or SearchService()
        yield
        if service is None:
            app.state.search.close()

    app = FastAPI(title="Local Hybrid Search", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Prevent other websites from triggering local indexing or opening files.
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin requests are disabled."}, status_code=403)
        if request.method == "POST" and "application/json" not in request.headers.get("content-type", ""):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "JSON request required."}, status_code=415)
        return await call_next(request)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return app.state.search.status()

    @app.post("/api/index")
    def index(body: IndexRequest):
        try:
            return app.state.search.index_folder(body.folder)
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/search")
    def search(body: SearchRequest):
        try:
            return app.state.search.search(**body.model_dump())
        except (ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/files/{file_id}/open")
    def open_file(file_id: int):
        try:
            path = app.state.search.source(file_id)
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
            return {"opened": True}
        except OSError as error:
            raise HTTPException(404, str(error)) from error

    return app


app = create_app()
