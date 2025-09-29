from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Request Models ---

class SingleQueryRequest(BaseModel):
    """Input model for POST /api/classify"""
    text: str = Field(..., example="Can you book a meeting with the client next week?")

class BatchQueryRequest(BaseModel):
    """Input model for POST /api/classify/batch"""
    texts: List[str] = Field(..., example=["Find me information about the nearest coffee shop", "Send an email to HR"])

# --- Response Models ---

class SingleClassificationResponse(BaseModel):
    """Output model for POST /api/classify"""
    intent: str = Field(..., example="calendar_schedule")
    confidence: float = Field(..., example=0.98, description="Model confidence in the predicted intent.")

class BatchClassificationResult(BaseModel):
    """Item model for batch classification output"""
    text: str = Field(..., example="Send an email to HR")
    intent: str = Field(..., example="email_send")
    confidence: float = Field(..., example=0.92)


class ModelInfoResponse(BaseModel):
    """Output model for GET /api/model/info"""
    model_name: str = Field(..., example="Scikit-learn Logistic Regression Pipeline")
    version: str = Field(..., example="1.0.0")
    model_metrics: Dict[str, Any] = Field(
        ..., 
        example={
            "overall_accuracy": 0.9000,
            "macro_f1_score": 0.8964,
            "best_hyperparameters": {
                "classifier__C": 10
            }
        }
    )
    supported_intents: List[str] = Field(..., example=['email_send', 'calendar_schedule', 'web_search', 'knowledge_query', 'general_chat'])
    last_trained: str = Field(..., example="2025-09-29")