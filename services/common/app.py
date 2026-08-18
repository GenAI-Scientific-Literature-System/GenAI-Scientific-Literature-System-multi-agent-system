from fastapi import FastAPI


def service_app(name: str) -> FastAPI:
    app = FastAPI(title=f"GLAS-Med {name}", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": name}

    return app
