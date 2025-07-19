"""
Iris Classification API

This module provides a FastAPI-based REST API for classifying iris flowers
using a trained Random Forest model. The API accepts iris flower measurements
and returns species predictions with confidence probabilities.

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# Global variable to store the loaded machine learning model
model = None

# Mapping from numeric predictions to species names
# 0: Iris Setosa, 1: Iris Versicolor, 2: Iris Virginica
label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

def load_model():
    """
    Load the trained Random Forest model.
    
    Load the model from multiple possible file paths to handle
    different deployment environments (local, Docker, cloud platforms).
    
    Returns:
        bool: True if model loaded successfully, False otherwise.
        
    Side Effects:
        Sets the global 'model' variable if successful.
        Prints diagnostic information about file system and loading attempts.
    """
    global model
    
    # List of possible model file locations for different deployment scenarios
    possible_paths = [
        "app/model/trained_rf_model.pkl",    # Standard Docker/local structure
        "./app/model/trained_rf_model.pkl",  # Alternative relative path (IBM Cloud)
        "trained_rf_model.pkl"               # Model moved to root directory
    ]
    
    # Get and display current working directory for debugging
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Try each possible path until one works
    for MODEL_PATH in possible_paths:
        print(f"Trying model path: {os.path.abspath(MODEL_PATH)}")
        try:
            # Check if file exists before attempting to load
            if os.path.exists(MODEL_PATH):
                # Load the pickled model using joblib
                model = joblib.load(MODEL_PATH)
                print(f"Model loaded successfully from: {os.path.abspath(MODEL_PATH)}")
                return True
        except Exception as e:
            # Log any loading errors and continue to next path
            print(f"Failed to load from {MODEL_PATH}: {e}")
            continue
    
    # If no model was found, provide detailed debugging information
    print("Model file not found in any expected location")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # List contents of expected directories if they exist
    if os.path.exists("app"):
        print(f"Files in app/: {os.listdir('app/')}")
    if os.path.exists("app/model"):
        print(f"Files in app/model/: {os.listdir('app/model/')}")
    
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    
    Handles application initialization (model loading) during startup
    and cleanup during shutdown.
    
    Args:
        app (FastAPI): The FastAPI application instance.
        
    Yields:
        None: Control to the application during its lifetime.
    """
    # === STARTUP ===
    print("Starting Iris Classification API...")
    success = load_model()
    
    if success:
        print("API ready to serve predictions!")
    else:
        print("WARNING: API started but model failed to load!")
    
    # Yield control to the application
    yield
    
    # === SHUTDOWN ===
    print("Shutting down Iris Classification API...")

# Initialize FastAPI application with metadata and lifespan management
app = FastAPI(
    title="Iris Classification API",
    description="A machine learning API for classifying iris flowers based on sepal and petal measurements",
    version="1.0",
    lifespan=lifespan
)

class IrisData(BaseModel):
    """
    Pydantic model for input validation of iris flower measurements.
    
    All measurements are expected to be in centimeters and must be positive floats.
    
    Attributes:
        sepal_length (float): Length of the sepal in cm (typically 4.0-8.0)
        sepal_width (float): Width of the sepal in cm (typically 2.0-4.5)
        petal_length (float): Length of the petal in cm (typically 1.0-7.0)
        petal_width (float): Width of the petal in cm (typically 0.1-2.5)
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        """Pydantic configuration for the model."""
        # Example values for API documentation
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (restrict in prod)
    allow_credentials=True,       # Allow cookies and authorization headers
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

@app.post("/predict")
def predict(iris_data: IrisData):
    """
    Predict the iris species based on flower measurements.
    
    This endpoint accepts iris flower measurements and returns the predicted
    species along with confidence probabilities for each possible class.
    
    Args:
        iris_data (IrisData): Iris flower measurements (sepal/petal dimensions)
        
    Returns:
        dict: Prediction results containing:
            - species (str): Predicted iris species name
            - prediction_class (int): Numeric class (0=setosa, 1=versicolor, 2=virginica)
            - probabilities (dict): Confidence scores for each species
            
    Raises:
        HTTPException: 
            - 500 if model is not loaded
            - 500 if prediction fails due to invalid input or model error
            
    Example:
        Input: {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        Output: {
            "species": "setosa",
            "prediction_class": 0,
            "probabilities": {"setosa": 0.95, "versicolor": 0.03, "virginica": 0.02}
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded - server configuration error"
            )
        
        # Convert input data to numpy array format expected by the model
        # Shape: (1, 4) - one sample with four features
        features = np.array([
            [
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width
        ]])
        
        # Generate prediction using the loaded model
        prediction = model.predict(features)[0]           # Get class prediction (0, 1, or 2)
        prediction_proba = model.predict_proba(features)[0]  # Get probability distribution
        
        # Convert numeric prediction to human-readable species name
        species = label_map[prediction]
        
        # Construct comprehensive response with all relevant information
        response = {
            "species": species,                           # Human-readable species name
            "prediction_class": int(prediction),          # Numeric class for programmatic use
            "probabilities": {                           # Confidence scores for each class
                "setosa": float(prediction_proba[0]),
                "versicolor": float(prediction_proba[1]),
                "virginica": float(prediction_proba[2])
            }
        }
        
        return response
        
    except Exception as e:
        # Log the error and return HTTP 500 with error details
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/")
def read_root():
    """
    Root endpoint providing basic API information and status.
    
    Returns:
        dict: API information including name and model loading status
    """
    return {
        "message": "Iris Classification API",
        "description": "Machine learning API for iris species classification",
        "status": "Model loaded" if model else "Model not loaded",
        "docs_url": "/docs",
        "version": "1.0"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring and load balancer integration.
    
    This endpoint is commonly used by:
    - Load balancers to determine if the service is healthy
    - Monitoring systems to track service availability
    - Container orchestration platforms (Docker, Kubernetes)
    
    Returns:
        dict: Health status information including:
            - status: Overall health status
            - model_loaded: Whether the ML model is loaded and ready
            - api_version: Current API version
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0",
        "timestamp": "2025-07-20" # Example static timestamp, can be dynamic if needed
    }

# Main execution block - runs when script is executed directly
if __name__ == "__main__":

    # Configuration for different deployment environments
    # Get port from environment variable (required for many cloud platforms)
    port = int(os.environ.get("PORT", 8000))
    
    # Use 0.0.0.0 to accept connections from any IP (required for containerized deployments)
    host = "0.0.0.0"  
    
    # Display startup information
    print(f"Starting API server on {host}:{port}")
    print("Visit the /docs endpoint for interactive API documentation")
    print("Visit the /health endpoint for health checks")
    
    # Start the Uvicorn ASGI server
    uvicorn.run(
        "app.api:app",    # Module and app instance
        host=host,        # Bind address
        port=port,        # Bind port
        reload=False      # Disable auto-reload for prod (True for dev)
    )