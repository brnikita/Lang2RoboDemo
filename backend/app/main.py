"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.capture import router as capture_router

__all__ = ["app"]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with CORS middleware and routes.
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
    return application


app = create_app()
