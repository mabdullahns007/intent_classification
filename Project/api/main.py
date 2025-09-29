from fastapi import FastAPI
from api.endpoints import router
from ml.model_loader import get_model_loader
from contextlib import asynccontextmanager
import time

# --- Lifespan Context Manager (Model Loading) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. Loads the ML model on startup.
    """
    start_time = time.time()
    try:
        model_loader = get_model_loader()
        model_loader.load_model() # Model loading happens here (caching)
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load the ML model. The API will be in a degraded state (503 errors). Error: {e}")
        
    yield # Application serves requests
    
    # Shutdown logic (runs after yield)
    print("Application shutdown complete.")


# --- FastAPI Application Initialization ---

app = FastAPI(
    title="Intent Classification Backend",
    description="A FastAPI application serving a trained intent classification model. "
                "See `/docs` for auto-generated documentation.",
    version="1.0.0",
    lifespan=lifespan # Model loading on startup
)

# Include the API router with all defined endpoints
app.include_router(router)

# Root Endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Intent Classification API is running. See /docs for endpoints."}