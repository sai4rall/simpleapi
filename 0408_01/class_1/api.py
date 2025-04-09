from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Optional
from logic import AnomalyDetector
import uvicorn

class QueryRequest(BaseModel):
    services: Optional[Dict[str, Dict[str, str]]] = None
    namespace: Optional[str] = None
    retrain: bool = False
    contamination: Optional[float] = None

class FetchMetricsRequest(BaseModel):
    query: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    step: str = "1m"

class PredictFromCSVRequest(BaseModel):
    namespace: Optional[str] = None

class Api:
    def __init__(self):
        self.app = FastAPI(
            title="Anomaly Detection and RCA API",
            description="API for detecting anomalies and analyzing cascading failures within a single namespace with persistent models.",
            version="1.0.0"
        )
        self.detector = AnomalyDetector()

        # Define endpoints
        self.app.post("/train")(self.train_model)
        self.app.post("/predict")(self.predict_anomalies)
        self.app.post("/predict_from_csv")(self.predict_from_csv)
        self.app.get("/graph")(self.get_graph_image)
        self.app.get("/issue_graph")(self.get_issue_graph_image)
        self.app.get("/metric_graph")(self.get_metric_graph_image)
        self.app.post("/fetch_metrics")(self.fetch_metrics)

    async def train_model(self, request: QueryRequest) -> Dict[str, Any]:
        """Train per-job models."""
        try:
            services = request.services if request.services is not None else self.detector.default_services
            namespace = request.namespace if request.namespace is not None else self.detector.default_namespace
            contamination = request.contamination if request.contamination is not None else self.detector.default_contamination
            return self.detector.train_model(services, namespace, request.retrain, contamination)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    async def predict_anomalies(self, request: QueryRequest, enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        """Detect anomalies and generate expected behavior/deviation graphs."""
        try:
            services = request.services if request.services is not None else self.detector.default_services
            namespace = request.namespace if request.namespace is not None else self.detector.default_namespace
            return self.detector.predict_anomalies(services, namespace, enable_2min_duration, enable_5min_duration)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def predict_from_csv(self, request: PredictFromCSVRequest, file: UploadFile = File(...), enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        """Predict anomalies from a CSV file."""
        try:
            namespace = request.namespace if request.namespace is not None else self.detector.default_namespace
            return self.detector.predict_from_csv(namespace, file.file, enable_2min_duration, enable_5min_duration)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def get_graph_image(self) -> Response:
        """Serve the dependency graph image."""
        try:
            image_data = self.detector.get_graph_image()
            return Response(content=image_data, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def get_issue_graph_image(self) -> Response:
        """Serve the issue graph image."""
        try:
            image_data = self.detector.get_issue_graph_image()
            return Response(content=image_data, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def get_metric_graph_image(self, metric: str) -> Response:
        """Serve the metric behavior graph image."""
        try:
            namespace = self.detector.default_namespace  # Could be made dynamic
            image_data = self.detector.get_metric_graph_image(metric, namespace)
            return Response(content=image_data, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def fetch_metrics(self, request: FetchMetricsRequest) -> Dict[str, Any]:
        """Fetch Prometheus metrics using a custom query."""
        try:
            return self.detector.fetch_metrics(request.query, request.start_time, request.end_time, request.step)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI application."""
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    api = Api()
    api.run()
