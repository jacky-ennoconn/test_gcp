from fastapi import FastAPI, Request
import joblib
import numpy as np

from .routes.v1 import user

app = FastAPI(
    title="Cloud Run API Demo", description="", version="0.1.0", redoc_url=None
)

# Load the pre-trained XGBoost model
model = joblib.load("xgboost_model.pkl")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def hello():
    return {"hello": "world"}

# Endpoint to make predictions using the XGBoost model
@app.post("/predict/")
async def predict(input_data: list):
    # Convert the input data to a numpy array
    input_array = np.array(input_data)
    # Make predictions
    predictions = model.predict(input_array.reshape(1, -1))
    return {"predictions": predictions.tolist()}

# Include the user router
app.include_router(user.router, prefix="/api/v1/users", tags=["User"])
