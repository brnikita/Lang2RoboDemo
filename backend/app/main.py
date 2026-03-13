"""FastAPI application entrypoint."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.capture import router as capture_router
from backend.app.api.iterate import router as iterate_router
from backend.app.api.recommend import router as recommend_router
from backend.app.api.simulate import router as simulate_router

__all__ = ["app"]

_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with CORS, routes, and static files.
    """
    application = FastAPI(
        title="Lang2Robo",
        description="Text description → robotic cell simulation → iterative improvement",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(capture_router)
    application.include_router(recommend_router)
    application.include_router(simulate_router)
    application.include_router(iterate_router)

    _mount_frontend(application)

    return application


def _mount_frontend(application: FastAPI) -> None:
    """Mount built frontend if available, with SPA fallback.

    Args:
        application: FastAPI instance.
    """
    if not _FRONTEND_DIST.exists():
        return

    application.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="static",
    )

    @application.get("/{full_path:path}")
    async def spa_fallback(request: Request, full_path: str) -> FileResponse:
        """Serve index.html for all non-API routes (SPA routing).

        Args:
            request: HTTP request.
            full_path: Requested path.

        Returns:
            index.html file response.
        """
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")


app = create_app()
