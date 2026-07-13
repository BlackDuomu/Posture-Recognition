"""Start the local model service with the required single worker."""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("MODEL_HOST", "127.0.0.1"),
        port=int(os.getenv("MODEL_PORT", "8000")),
        workers=1,
    )
