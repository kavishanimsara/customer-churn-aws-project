"""Application entrypoint: FastAPI REST API with the Gradio UI mounted at ``/``."""

from __future__ import annotations

import argparse

import gradio as gr
import uvicorn
from fastapi import FastAPI

from .api import router
from .ui import build_ui


def create_app() -> FastAPI:
    app = FastAPI(
        title="Customer Churn API",
        version="0.1.0",
        description="Churn predictions and model artifacts for the churn pipeline.",
    )
    app.include_router(router)
    app = gr.mount_gradio_app(app, build_ui(), path="/")
    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the churn dashboard server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
