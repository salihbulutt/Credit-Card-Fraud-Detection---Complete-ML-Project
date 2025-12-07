"""
FastAPI REST API for Credit Card Fraud Detection
Provides RESTful endpoints for fraud prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import FraudDetector
from src.config import API_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector globally
try:
    detector = FraudDetector()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# Pydantic Models
class Transaction(BaseModel):
    """Single transaction data model"""
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Time: float = Field(0.0, description="Seconds elapsed since first transaction", ge=0)
    Amount: float = Field(..., description="Transaction amount", ge=0)
    
    @validator('Amount')
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        if v > 100000:
            raise ValueError('Amount exceeds maximum allowed value')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "V1": -1.359807134,
                "V2": -0.072781173,
                "V3": 2.536346738,
                "V4": 1.378155224,
                "V5": -0.338320770,
                "V6": 0.462387778,
                "V7": 0.239598554,
                "V8": 0.098697901,
                "V9": 0.363786970,
                "V10": 0.090794172,
                "V11": -0.551599533,
                "V12": -0.617800856,
                "V13": -0.991389847,
                "V14": -0.311169354,
                "V15": 1.468176972,
                "V16": -0.470400525,
                "V17": 0.207971242,
                "V18": 0.025790720,
                "V19": 0.403992960,
                "V20": 0.251412098,
                "V21": -0.018306778,
                "V22": 0.277837576,
                "V23": -0.110473910,
                "V24": 0.066928075,
                "V25": 0.128539358,
                "V26": -0.189114844,
                "V27": 0.133558377,
                "V28": -0.021053053,
                "Time": 0,
                "Amount": 149.62
            }
        }

class BatchTransactions(BaseModel):
    """Batch of transactions"""
    transactions: List[Transaction] = Field(..., description="List of transactions to predict")
    
    @validator('transactions')
    def check_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Maximum batch size is 1000 transactions')
        if len(v) == 0:
            raise ValueError('At least one transaction is required')
        return v

class PredictionResponse(BaseModel):
    """Prediction response model"""
    transaction_id: Optional[str] = None
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: float
    threshold_used: float
    recommendation: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_transactions: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
    timestamp: str

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": API_CONFIG["version"],
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": "1.0.0",
        "api_version": API_CONFIG["version"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    - **transaction**: Transaction data with all required features (V1-V28, Time, Amount)
    
    Returns fraud prediction with probability, risk level, and recommendation.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {model_error}"
        )
    
    try:
        start_time = datetime.now()
        
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Validate transaction
        validation = detector.validate_transaction(transaction_dict)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"errors": validation['errors']}
            )
        
        # Make prediction
        result = detector.predict_with_details(transaction_dict)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTransactions):
    """
    Predict fraud for multiple transactions
    
    - **transactions**: List of transactions (max 1000)
    
    Returns predictions for all transactions with summary statistics.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {model_error}"
        )
    
    try:
        start_time = datetime.now()
        
        # Convert to list of dicts
        transactions_list = [t.dict() for t in batch.transactions]
        
        # Make predictions
        results = []
        fraud_count = 0
        
        for idx, transaction in enumerate(transactions_list):
            result = detector.predict_with_details(transaction)
            result['transaction_id'] = f"txn_{idx}"
            result['timestamp'] = datetime.now().isoformat()
            results.append(result)
            if result['is_fraud']:
                fraud_count += 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(results) if results else 0,
            "predictions": results,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )

@app.post("/explain", tags=["Explanation"])
async def explain_prediction(transaction: Transaction):
    """
    Explain why a prediction was made
    
    - **transaction**: Transaction data to explain
    
    Returns feature importance and contribution to the prediction.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {model_error}"
        )
    
    try:
        transaction_dict = transaction.dict()
        explanation = detector.explain_prediction(transaction_dict)
        return explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation error: {str(e)}"
        )

@app.post("/validate", tags=["Validation"])
async def validate_transaction(transaction: Transaction):
    """
    Validate transaction data without making a prediction
    
    - **transaction**: Transaction data to validate
    
    Returns validation results with any errors or warnings.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {model_error}"
        )
    
    try:
        transaction_dict = transaction.dict()
        validation = detector.validate_transaction(transaction_dict)
        return validation
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information and statistics"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {model_error}"
        )
    
    return {
        "model_type": "XGBoost",
        "model_version": "1.0.0",
        "features_count": len(detector.feature_names),
        "feature_names": detector.feature_names,
        "threshold": detector.model.predict(np.zeros((1, len(detector.feature_names))))[0],
        "model_loaded": True
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True
    )
