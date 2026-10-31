import os

from fastapi import FastAPI


if os.getenv("DEMO_MODE") == "1" and os.getenv("USE_FIXTURES") == "1":
    print("ERROR: DEMO_MODE=1 and USE_FIXTURES=1 are not allowed together.")
    raise SystemExit(1)

app = FastAPI(title="BreakWatch")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
