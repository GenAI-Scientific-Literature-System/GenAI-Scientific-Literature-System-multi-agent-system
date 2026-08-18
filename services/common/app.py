try:
    from fastapi import FastAPI
except ImportError:
    class FastAPI:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs):
            def decorator(fn): return fn
            return decorator
        def post(self, *args, **kwargs):
            def decorator(fn): return fn
            return decorator


def service_app(name: str) -> FastAPI:
    app = FastAPI(title=f"GLAS-Med {name}", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": name}

    return app
