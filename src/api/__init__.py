def app():
    from src.api.main import app
    return app

__all__ = ["app"]