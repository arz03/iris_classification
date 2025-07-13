# This file is part of the Iris Classification project.
# It contains the API endpoints for model predictions.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# Global variable to store the model
model = None

# Label mapping
label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

def load_model():
    """Load the trained model from disk."""
    global model
    # Path for Docker container
    MODEL_PATH = "app/model/trained_rf_model.pkl"
    
    # Add debug information
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file does not exist at: {os.path.abspath(MODEL_PATH)}")
            # List files for debugging
            if os.path.exists("app/model/"):
                print(f"Files in app/model/: {os.listdir('app/model/')}")
            return False
            
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from: {os.path.abspath(MODEL_PATH)}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Iris Classification API...")
    success = load_model()
    if success:
        print("API ready to serve predictions!")
    else:
        print("WARNING: API started but model failed to load!")
    
    yield
    
    # Shutdown
    print("Shutting down Iris Classification API...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Iris Classification API", version="1.0", lifespan=lifespan)

# Pydantic model for input validation
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(iris_data: IrisData):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert input to numpy array
        features = np.array([
            [
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Convert prediction to species name
        species = label_map[prediction]
        
        # Create response with probabilities
        response = {
            "species": species,
            "prediction_class": int(prediction),
            "probabilities": {
                "setosa": float(prediction_proba[0]),
                "versicolor": float(prediction_proba[1]),
                "virginica": float(prediction_proba[2])
            }
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Iris Classification API", "status": "Model loaded" if model else "Model not loaded"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "1.0"
    }

if __name__ == "__main__":
    # Run on localhost:8000 with reload enabled
    print("Starting API server on http://localhost:8000")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)