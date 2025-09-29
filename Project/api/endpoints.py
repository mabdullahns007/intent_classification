from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from api.models import (
    SingleQueryRequest, 
    SingleClassificationResponse,
    BatchQueryRequest,
    BatchClassificationResult,
    ModelInfoResponse
)
from ml.model_loader import get_model_loader, IntentClassifier

router = APIRouter(prefix="/api")
security = HTTPBasic()

# --- Basic Authentication Configuration ---

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secretpassword" 

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticates admin credentials using HTTP Basic Auth."""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        # Error handling and appropriate HTTP status code (401 Unauthorized)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- GET /api/health - Health Check Endpoint ---
@router.get(
    "/health",
    tags=["Utility"],
    summary="Health check endpoint",
    status_code=status.HTTP_200_OK
)
async def health_check():
    """Returns the status of the API and underlying ML model."""
    model_loader = get_model_loader()
    
    status_detail = {
        "api_status": "ok",
        "model_loaded": model_loader.is_loaded
    }
    
    if not model_loader.is_loaded:
        # Returns 503 Service Unavailable if the model failed to load
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail={"error": "ML model not loaded.", **status_detail}
        )
        
    return status_detail

# --- POST /api/classify - Classify single query ---
@router.post(
    "/classify", 
    response_model=SingleClassificationResponse,
    tags=["Classification"],
    summary="Classify a single user query",
    status_code=status.HTTP_200_OK
)
async def classify_single_query(
    request: SingleQueryRequest,
    model_loader: IntentClassifier = Depends(get_model_loader)
):
    """Classifies a single text string to predict the user's intent."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML model is not loaded and ready for inference."
        )

    try:
        intent, confidence = model_loader.classify_single(request.text)
        return SingleClassificationResponse(intent=intent, confidence=confidence)
    except Exception as e:
        # Generic 500 Internal Server Error for unexpected classification failures
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed due to an internal error: {e}"
        )

# --- POST /api/classify/batch - Classify multiple queries ---
@router.post(
    "/classify/batch", 
    response_model=list[BatchClassificationResult],
    tags=["Classification"],
    summary="Classify multiple user queries in a batch",
    status_code=status.HTTP_200_OK
)
async def classify_batch_queries(
    request: BatchQueryRequest,
    model_loader: IntentClassifier = Depends(get_model_loader)
):
    """Classifies a list of text strings and returns a list of results."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML model is not loaded and ready for inference."
        )
    
    if not request.texts:
        return []

    try:
        results = model_loader.classify_batch(request.texts)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed due to an internal error: {e}"
        )


# --- GET /api/model/info - Return model metadata and performance metrics (Auth required) ---
@router.get(
    "/model/info", 
    response_model=ModelInfoResponse,
    tags=["Model Management"],
    summary="Return model metadata and performance metrics (Requires Basic Auth)",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(authenticate_admin)]
)
async def get_model_info():
    """
    Retrieves metadata, configuration, and performance metrics for the deployed model.
    Requires Basic Authentication (Username: 'admin', Password: 'secretpassword').
    """
    model_loader = get_model_loader()
    
    if not model_loader.is_loaded:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded. Cannot retrieve metadata."
        )

    # Metrics and Intents extracted from the provided Notebook
    MODEL_METRICS = {
        "overall_accuracy": 0.9000,
        "macro_f1_score": 0.8964,
        "best_hyperparameters": {
            "classifier__C": 10,
            "classifier__solver": "liblinear",
            "tfidf__min_df": 1,
            "tfidf__ngram_range": "(1, 1)"
        },
    }
    
    SUPPORTED_INTENTS = [
        'calendar_schedule', 'email_send', 'general_chat', 
        'knowledge_query', 'web_search'
    ]

    model_info = ModelInfoResponse(
        model_name="Scikit-learn Logistic Regression Pipeline",
        version="1.0.0",
        model_metrics=MODEL_METRICS,
        supported_intents=SUPPORTED_INTENTS,
        last_trained="2025-09-29" 
    )
    
    return model_info