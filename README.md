

````markdown
# 🤖 Intent Classification FastAPI Backend

This project implements a **FastAPI backend** to serve a trained **scikit-learn intent classification model**. It is designed with production-ready features, including health checks, efficient batch processing, and authenticated model information retrieval.

---

## 🚀 Setup and Running

This guide covers both local development and containerized deployment using Docker.

### Prerequisites

* **Docker** (Recommended for deployment)
* **Python 3.9+** and **pip** (For local development)

### 1. Local Development Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Server:**
    The application automatically loads the pre-trained model (from `.pkl` files in the `ml/` directory) during startup.
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`.

---

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

---

### 3. Running Unit Tests

To ensure all endpoints are functioning correctly and the model loads as expected, run the provided tests:

```bash
pytest tests/test_api.py
````

-----

## 🧭 API Endpoints

### ✅ Model Status & Classification

| Endpoint | Method | Description | Authentication |
| :--- | :--- | :--- | :--- |
| **`/api/health`** | `GET` | **Health Check.** Returns `200 OK` if the server is running and the intent classification model has successfully loaded into memory. | None |
| **`/api/classify`** | `POST` | **Classify a single user query.** | None |
| **`/api/classify/batch`** | `POST` | **Classify multiple user queries simultaneously.** | None |
| **`/api/model/info`** | `GET` | **Retrieve model metadata and performance metrics.** Provides information like accuracy, F1-score, hyper-parameters, and supported intents. | Basic Auth |

### Request and Response Schemas

#### **POST `/api/classify`**

  * **Input Body:**
    ```json
    {
      "text": "user query"
    }
    ```
  * **Output Body:**
    ```json
    {
      "intent": "predicted_intent", 
      "confidence": 0.95
    }
    ```

#### **POST `/api/classify/batch`**

  * **Input Body:**
    ```json
    {
      "texts": ["query1", "query2", "query3"]
    }
    ```
  * **Output Body (List of results):**
    ```json
    [
      {"intent": "intent_a", "confidence": 0.91},
      {"intent": "intent_b", "confidence": 0.88},
      // ... more results
    ]
    ```

-----

## 🔐 Authentication

The **Model Management Endpoint (`/api/model/info`)** is secured with **HTTP Basic Authentication**. You must provide the following credentials in the request header:

| Attribute | Value |
| :--- | :--- |
| **Username** | `admin` |
| **Password** | `secretpassword` |

```
```
