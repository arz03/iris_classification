# 🌸 Iris Classification API

A machine learning project that classifies iris flowers using a Random Forest model, deployed as a FastAPI web service.

## 🚀 Features

- **High Accuracy**: 98%+ accuracy on test data
- **REST API**: FastAPI-based web service
- **Interactive UI**: HTML interface for testing
- **Docker Support**: Containerized deployment
- **Real-time Predictions**: Instant species classification

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: 100%
- **Testing Accuracy**: 98%
- **Dataset**: Iris flower dataset (150 samples, 4 features)

## 📁 Project Structure

```
iris_classification/
├── app/
│   ├── api.py                    # FastAPI application
│   └── model/
│       └── trained_rf_model.pkl  # Trained ML model
├── model.ipynb                   # Model training notebook
├── test_interface.html           # Test interface
├── requirements.txt              # Development dependencies
├── requirements-prod.txt         # Production dependencies (minimal)
├── Dockerfile                    # Container configuration
├── manifest.yml                  # Cloud deployment configuration
└── README.md                     # This file
```

## 🛠️ Setup & Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/arz03/iris_classification.git
   cd iris_classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)
   ```bash
   jupyter notebook model.ipynb
   ```
   Note: Make sure the directory "iris_classification/app/model" is present before saving the model

5. **Run the API**
   ```bash
   cd app
   uvicorn api:app --reload --host localhost --port 8000
   ```

### Docker Deployment

```bash
# Build the image
docker build -t iris-classification .

# Run the container
docker run -p 8000:8000 iris-classification
```

**Note**: The Docker build uses `requirements-prod.txt` for a smaller, optimized production image. For development, use `requirements.txt` which includes additional tools like Jupyter.

## 🌐 API Usage

### Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /predict` - Make predictions

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

### Example Response

```json
{
  "species": "setosa",
  "prediction_class": 0,
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

## 🎯 Testing

Open [test_interface.html](test_interface.html) in your browser for an interactive testing interface.

## 📋 Dependencies

- **`requirements.txt`**: Full development environment with Jupyter, debugging tools, and all dependencies
- **`requirements-prod.txt`**: Minimal production dependencies for Docker deployment (smaller image size)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
