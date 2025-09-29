# Intent Classification FastAPI Backend

This project implements a **FastAPI backend** to serve a trained scikit-learn intent classification model, including essential features like health checks, batch processing, and authenticated model information retrieval.

## 📦 Project Structure
project/
├── ml/
│   ├── intent_classifier_pipeline.pkl  # Trained Model
│   ├── intent_label_encoder.pkl        # Label Encoder
│   └── model_loader.py                 # Logic for loading and running the model
├── api/
│   ├── main.py                         # FastAPI App Initialization & Lifespan
│   ├── models.py                       # Pydantic Schemas (Input/Output validation)
│   └── endpoints.py                    # API Routes (classify, info, health)
├── Dockerfile                          # Containerization instructions
├── tests/
│   └── test_api.py                     # Unit tests
└── requirements.txt                    # Python dependencies

## 🚀 Setup and Running

### Prerequisites

* **Docker** (Recommended for deployment)
* **Python 3.9+** and **pip** (For local development)

### 1. Local Development Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Server:**
    The application loads the model (`.pkl` files from `ml/`) automatically during startup.
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`.

### 2. Containerization with Docker

1.  **Build the Docker Image:**
    ```bash
    docker build -t intent-classifier-api .
    ```
2.  **Run the Container:**
    ```bash
    docker run -d --name intent-api -p 8000:8000 intent-classifier-api
    ```
    The API will be accessible at `http://localhost:8000`.

### 3. Running Unit Tests

To ensure all endpoints are functioning correctly, run the provided tests:

```bash
pytest tests/test_api.py

API Endpoints

### ✅ Intent Classification API Endpoints

* **GET `/api/health`**
    * **Description**: Performs a health check.
    * **Purpose**: Returns `200 OK` if the API server is running and the intent classification model has successfully loaded into memory.
    * **Authentication**: None.

---

* **POST `/api/classify`**
    * **Description**: Classify a single user query.
    * **Input**: JSON body: `{"text": "user query"}`.
    * **Output**: JSON body: `{"intent": "predicted_intent", "confidence": 0.95}`.
    * **Authentication**: None.

---

* **POST `/api/classify/batch`**
    * **Description**: Classify multiple user queries simultaneously.
    * **Input**: JSON body: `{"texts": ["query1", "query2"]}`.
    * **Output**: A JSON list of classification results, with an entry for each query.
    * **Authentication**: None.

---

* **GET `/api/model/info`**
    * **Description**: Retrieve model metadata and performance metrics.
    * **Purpose**: Provides information like accuracy, F1-score, hyper-parameters, and supported intents.
    * **Authentication**: **Basic Auth** (Username: `admin`, Password: `secretpassword`).
🔐 Authentication
The Model Management Endpoint (/api/model/info) is secured with HTTP Basic Authentication.

Username: admin

Password: secretpassword