project/
│
├── api.py                # FastAPI endpoints
├── anomaly_detector.py   # Business logic class
├── main.py               # Main application entry point
└── config/               # Configuration files (prometheus_queries.json, service_names.json)
    ├── prometheus_queries.json
    └── service_names.json


# anomaly_detector.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from scipy.signal import correlate
from prometheus_api_client import PrometheusConnect
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import pystache
import logging

class AnomalyDetector:
    def __init__(self, prometheus_url: str, proxy_url: str, proxy_username: Optional[str], proxy_password: Optional[str],
                 model_dir: str, graph_dir: str, data_dir: str, prediction_dir: str, graph_image_path: str,
                 issue_graph_image_path: str, query_config_path: str, service_config_path: str,
                 services: int, default_namespace: str, default_contamination: float, outlier_trim_percentile: float):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.prometheus_url = prometheus_url
        self.proxy_url = proxy_url
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.model_dir = model_dir
        self.graph_dir = graph_dir
        self.data_dir = data_dir
        self.prediction_dir = prediction_dir
        self.graph_image_path = graph_image_path
        self.issue_graph_image_path = issue_graph_image_path
        self.query_config_path = query_config_path
        self.service_config_path = service_config_path
        self.services = services
        self.default_namespace = default_namespace
        self.default_contamination = default_contamination
        self.outlier_trim_percentile = outlier_trim_percentile

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Initialize Prometheus client
        proxy_dict = {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
        if self.proxy_username and self.proxy_password:
            proxy_dict["http"] = f"http://{self.proxy_username}:{self.proxy_password}@{self.proxy_url.split('://')[1]}"
            proxy_dict["https"] = f"https://{self.proxy_username}:{self.proxy_password}@{self.proxy_url.split('://')[1]}"
        self.client = PrometheusConnect(url=self.prometheus_url, disable_ssl=True, requests_params={"proxies": proxy_dict})

        # Load configurations
        with open(self.query_config_path, "r") as f:
            config = json.load(f)
            self.metrics = config["metrics"]

        with open(self.service_config_path, "r") as f:
            service_config = json.load(f)
            self.default_services = service_config["services"]
            if len(self.default_services) != self.services:
                raise ValueError(f"Expected {self.services} services in {self.service_config_path}, got {len(self.default_services)}")

    def get_model_path(self, namespace: str) -> str:
        return os.path.join(self.model_dir, f"models_{namespace}.pkl")

    def get_graph_path(self, namespace: str) -> str:
        return os.path.join(self.graph_dir, f"graph_{namespace}.pkl")

    def get_data_path(self, namespace: str) -> str:
        return os.path.join(self.data_dir, f"trained_data_{namespace}.csv")

    def get_prediction_path(self, namespace: str, timestamp: str) -> str:
        return os.path.join(self.prediction_dir, f"prediction_{namespace}_{timestamp}.csv")

    def fetch_prometheus_data(self, lookback: str, services: Dict[str, Dict[str, str]], namespace: str) -> pd.DataFrame:
        end_time = datetime.utcnow()
        if lookback == "2w":
            start_time = end_time - timedelta(days=14)
            step = "5m"
        elif lookback == "1h":
            start_time = end_time - timedelta(hours=1)
            step = "1m"
        else:
            raise ValueError("Invalid lookback period")

        data = {}
        for service_name, props in list(services.items())[:self.services]:
            context = {"service_name": service_name, "namespace": namespace, **props}
            for metric in self.metrics:
                applicable_services = metric.get("services", None)
                if applicable_services and service_name not in applicable_services:
                    continue
                query = pystache.render(metric["query"], context)
                try:
                    result = self.client.custom_query_range(
                        query=query,
                        start_time=start_time,
                        end_time=end_time,
                        step=step
                    )
                    self.logger.info(f"Query for {namespace}/{service_name}_{metric['name']}: {query}, Result: {len(result)} series")
                    
                    if not result:
                        self.logger.warning(f"No data returned for query: {query}")
                        data[f"{service_name}_{metric['name']}"] = pd.Series([], index=[])
                        continue

                    for series in result:
                        labels = series.get("metric", {})
                        labels.pop("__name__", None)
                        label_str = "_".join([f"{k}={v}" for k, v in sorted(labels.items())])
                        col_name = f"{service_name}_{metric['name']}"
                        if label_str:
                            col_name += f"_{label_str}"

                        if series["values"]:
                            timestamps = [datetime.fromtimestamp(float(t)) for t, _ in series["values"]]
                            values = [float(v) for _, v in series["values"]]
                            data[col_name] = pd.Series(values, index=timestamps)
                        else:
                            data[col_name] = pd.Series([], index=[])
                except Exception as e:
                    self.logger.error(f"Error fetching {metric['name']} for {namespace}/{service_name}: {str(e)}")
                    data[f"{service_name}_{metric['name']}"] = pd.Series([], index=[])

        df = pd.DataFrame(data)
        if df.empty:
            self.logger.warning(f"DataFrame is empty for {namespace}")
            return df

        df.index = pd.to_datetime(df.index)
        self.logger.info(f"DataFrame shape for {namespace}: {df.shape}")
        return df

    def remove_outliers(self, df: pd.DataFrame, percentile: float = None) -> pd.DataFrame:
        percentile = percentile if percentile is not None else self.outlier_trim_percentile
        if df.empty:
            self.logger.warning("DataFrame is empty, skipping outlier removal")
            return df

        self.logger.info(f"Removing top {percentile}% and bottom {percentile}% outliers from DataFrame")
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].notna().sum() == 0:
                continue
            lower_bound = df_cleaned[col].quantile(percentile / 100)
            upper_bound = df_cleaned[col].quantile(1 - (percentile / 100))
            df_cleaned[col] = df_cleaned[col].where(
                (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound), other=np.nan
            )
        
        df_cleaned = df_cleaned.dropna(how='all')
        self.logger.info(f"After removing outliers, DataFrame shape: {df_cleaned.shape}")
        return df_cleaned

    def preprocess_data(self, df: pd.DataFrame, remove_outliers_flag: bool = False) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("DataFrame is empty before preprocessing")
            return df

        df = df.resample("5T").mean().fillna(0)
        df = df.rolling(window=3, min_periods=1).mean()
        self.logger.info(f"After resampling, fillna, and smoothing, DataFrame shape: {df.shape}")
        
        df = df.loc[~(df == 0).all(axis=1)]
        self.logger.info(f"After filtering zero rows, DataFrame shape: {df.shape}")
        
        if df.empty:
            self.logger.warning("DataFrame is empty after preprocessing")
            return df
        
        if remove_outliers_flag:
            df = self.remove_outliers(df)
            if df.empty:
                self.logger.warning("DataFrame is empty after outlier removal")
                return df
        
        df = (df - df.mean()) / df.std()
        return df

    def infer_dependency_graph(self, df: pd.DataFrame, service_names: List[str]) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
        G = nx.DiGraph()
        lag_steps = 3
        service_columns = {service: [] for service in service_names}

        for col in df.columns:
            for service in service_names:
                if col.startswith(f"{service}_"):
                    service_columns[service].append(col)
                    break

        for service_a in service_names:
            for service_b in service_names:
                if service_a != service_b:
                    for metric_a in service_columns[service_a]:
                        for metric_b in service_columns[service_b]:
                            series_a = df.get(metric_a, pd.Series())
                            series_b = df.get(metric_b, pd.Series())
                            if not series_a.empty and not series_b.empty:
                                aligned = pd.concat([series_a, series_b], axis=1, join="inner")
                                if len(aligned) > lag_steps:
                                    corr = correlate(aligned.iloc[:, 0], aligned.iloc[:, 1], mode="full")
                                    lag_range = corr[len(aligned) - lag_steps:len(aligned)]
                                    max_corr = np.max(np.abs(lag_range))
                                    if max_corr > 0.7:
                                        G.add_edge(service_a, service_b, weight=max_corr)
                                        break
        self.logger.info(f"Dependency graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, service_columns

    def draw_graph(self, G: nx.DiGraph, namespace: str) -> Dict[str, Any]:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(f"Dependency Graph of Services in {namespace}")
        plt.savefig(self.graph_image_path, format="png", dpi=300)
        plt.close()
        edges = [{"source": src, "target": dst, "weight": data["weight"]} for src, dst, data in G.edges(data=True)]
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edge_list": edges}

    def draw_issue_graph(self, G: nx.DiGraph, namespace: str, root_cause: Optional[str], affected_services: List[str]) -> Dict[str, Any]:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        node_colors = []
        for node in G.nodes():
            if node == root_cause:
                node_colors.append("red")
            elif node in affected_services:
                node_colors.append("yellow")
            else:
                node_colors.append("lightblue")
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, arrows=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(f"Issue Graph of Services in {namespace} (Red: Root Cause, Yellow: Affected)")
        plt.savefig(self.issue_graph_image_path, format="png", dpi=300)
        plt.close()
        edges = [{"source": src, "target": dst, "weight": data["weight"]} for src, dst, data in G.edges(data=True)]
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edge_list": edges}

    def train_model(self, services: Optional[Dict[str, Dict[str, str]]], namespace: Optional[str], retrain: bool, contamination: Optional[float]) -> Dict[str, Any]:
        services = services if services is not None else self.default_services
        namespace = namespace if namespace is not None else self.default_namespace
        contamination = contamination if contamination is not None else self.default_contamination
        if len(services) < self.services:
            raise ValueError(f"Expected {self.services} services, got {len(services)}")
        service_names = list(services.keys())

        model_path = self.get_model_path(namespace)
        graph_path = self.get_graph_path(namespace)
        data_path = self.get_data_path(namespace)

        if retrain or not os.path.exists(model_path) or not os.path.exists(data_path):
            self.logger.info(f"Training: Fetching fresh data from Prometheus for {namespace}")
            df = self.fetch_prometheus_data(lookback="2w", services=services, namespace=namespace)
            if df.empty:
                raise ValueError(f"No data fetched from Prometheus for {namespace}")
            
            df = self.preprocess_data(df, remove_outliers_flag=True)
            if df.empty:
                raise ValueError(f"Preprocessed DataFrame is empty for {namespace} after outlier removal")
            
            df.to_csv(data_path)
            self.logger.info(f"Saved preprocessed data to {data_path}")

            models = {}
            G, service_columns = self.infer_dependency_graph(df, service_names)
            for service in service_names:
                service_cols = service_columns.get(service, [])
                if service_cols and not df[service_cols].dropna().empty:
                    try:
                        iso_forest = IsolationForest(
                            contamination=contamination,
                            n_estimators=200,
                            random_state=42
                        )
                        models[service] = iso_forest.fit(df[service_cols].dropna())
                        self.logger.info(f"Trained model for {service} with {len(df[service_cols].dropna())} samples in {namespace}, contamination={contamination}")
                    except ValueError as e:
                        self.logger.warning(f"Skipping {service} due to insufficient data: {str(e)}")
                else:
                    self.logger.warning(f"No valid data for {service} in {namespace}")

            graph_summary = self.draw_graph(G, namespace)

            with open(model_path, "wb") as f:
                pickle.dump(models, f)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
        else:
            self.logger.info(f"Loading existing preprocessed data from {data_path} for {namespace}")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if df.empty:
                raise ValueError(f"Loaded DataFrame from {data_path} is empty")
            G, service_columns = self.infer_dependency_graph(df, service_names)
            graph_summary = self.draw_graph(G, namespace)

        return {
            "status": "success",
            "message": f"Model trained or loaded, graph inferred, and image saved for {namespace}",
            "data_points": len(df),
            "features": len(df.columns),
            "graph_summary": graph_summary,
            "graph_image_url": "/graph"
        }

    def predict_anomalies(self, df: pd.DataFrame, services: Optional[Dict[str, Dict[str, str]]], namespace: Optional[str], enable_2min_duration: bool, enable_5min_duration: bool) -> Dict[str, Any]:
        services = services if services is not None else self.default_services
        namespace = namespace if namespace is not None else self.default_namespace
        if len(services) < self.services:
            raise ValueError(f"Expected {self.services} services, got {len(services)}")
        service_names = list(services.keys())

        model_path = self.get_model_path(namespace)
        graph_path = self.get_graph_path(namespace)

        if not os.path.exists(model_path) or not os.path.exists(graph_path):
            raise ValueError(f"Model not trained for {namespace}. Run /train first.")

        with open(model_path, "rb") as f:
            models = pickle.load(f)
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        df = self.preprocess_data(df, remove_outliers_flag=False)
        if df.empty:
            raise ValueError(f"Preprocessed DataFrame is empty for {namespace}")
        
        df = df.fillna(0)
        last_hour_df = df.tail(12)

        _, service_columns = self.infer_dependency_graph(df, service_names)

        anomalies: Dict[str, Dict[str, str]] = {}
        anomaly_predictions = {}
        for service in service_names:
            service_cols = service_columns.get(service, [])
            if service_cols and service in models:
                preds = models[service].predict(last_hour_df[service_cols])
                anomaly_indices = last_hour_df.index[preds == -1]

                for idx, col in enumerate(service_cols):
                    anomaly_predictions[f"{col}_anomaly"] = preds[:, idx]

                if not enable_2min_duration and not enable_5min_duration:
                    if len(anomaly_indices) > 0:
                        anomalies[service] = {}
                        for col in service_cols:
                            metric_name = col.split('_', 1)[1]
                            anomaly_mask = preds == -1
                            if anomaly_mask.any():
                                first_anomaly_time = last_hour_df.index[anomaly_mask][0]
                                anomalies[service][metric_name] = str(first_anomaly_time)

                elif enable_2min_duration:
                    if len(anomaly_indices) >= 2:
                        has_consecutive = False
                        for i in range(len(preds) - 1):
                            if preds[i] == -1 and preds[i + 1] == -1:
                                has_consecutive = True
                                break
                        if has_consecutive:
                            anomalies[service] = {}
                            for col in service_cols:
                                metric_name = col.split('_', 1)[1]
                                anomaly_mask = preds == -1
                                if anomaly_mask.any():
                                    first_anomaly_time = last_hour_df.index[anomaly_mask][0]
                                    anomalies[service][metric_name] = str(first_anomaly_time)

                elif enable_5min_duration:
                    if len(anomaly_indices) >= 5:
                        has_consecutive = False
                        for i in range(len(preds) - 4):
                            if all(preds[i:i+5] == -1):
                                has_consecutive = True
                                break
                        if has_consecutive:
                            anomalies[service] = {}
                            for col in service_cols:
                                metric_name = col.split('_', 1)[1]
                                anomaly_mask = preds == -1
                                if anomaly_mask.any():
                                    first_anomaly_time = last_hour_df.index[i] if i > 0 else last_hour_df.index[anomaly_mask][0]
                                    anomalies[service][metric_name] = str(first_anomaly_time)

        root_cause = None
        cascading_graph: Dict[str, Dict[str, Any]] = {}
        affected_services = []
        if anomalies:
            anomaly_times = {svc: min(pd.to_datetime(list(times.values()))) for svc, times in anomalies.items()}
            sorted_anomalies = sorted(anomaly_times.items(), key=lambda x: x[1])
            
            for svc, time in sorted_anomalies:
                preds = list(G.predecessors(svc))
                has_earlier_pred = any(pred in anomalies and anomaly_times[pred] < time for pred in preds)
                if not has_earlier_pred:
                    root_cause = svc
                    break
            
            if root_cause:
                cascading_graph[root_cause] = {
                    "first_anomaly": str(anomaly_times[root_cause]),
                    "metrics": anomalies[root_cause],
                    "affected_services": []
                }
                visited = set([root_cause])
                queue = [root_cause]
                while queue:
                    current = queue.pop(0)
                    for succ in G.successors(current):
                        if succ in anomalies and succ not in visited:
                            cascading_graph[current]["affected_services"].append({
                                "service": succ,
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ]
                            })
                            cascading_graph[succ] = {
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ],
                                "affected_services": []
                            }
                            visited.add(succ)
                            queue.append(succ)
                            affected_services.append(succ)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prediction_path = self.get_prediction_path(namespace, timestamp)
        prediction_df = last_hour_df.copy()
        for col, pred_values in anomaly_predictions.items():
            prediction_df[col] = pred_values
        prediction_df.to_csv(prediction_path)
        self.logger.info(f"Saved prediction data to {prediction_path}")

        issue_graph_summary = None
        if root_cause:
            issue_graph_summary = self.draw_issue_graph(G, namespace, root_cause, affected_services)

        return {
            "status": "success",
            "anomalies": anomalies,
            "root_cause": root_cause,
            "cascading_failures_graph": cascading_graph,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_file": prediction_path,
            "issue_graph_image_url": "/issue_graph" if issue_graph_summary else None,
            "last_hour_df": last_hour_df
        }

    def fetch_metrics(self, query: str, start_time: Optional[str], end_time: Optional[str], step: str) -> Dict[str, Any]:
        end_time_dt = datetime.utcnow() if end_time is None else datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        start_time_dt = end_time_dt - timedelta(hours=1) if start_time is None else datetime.fromisoformat(start_time.replace("Z", "+00:00"))

        result = self.client.custom_query_range(
            query=query,
            start_time=start_time_dt,
            end_time=end_time_dt,
            step=step
        )

        if not result or not result[0]["values"]:
            return {"status": "success", "message": "No data found", "data": []}

        timestamps = [datetime.fromtimestamp(float(t)).isoformat() for t, _ in result[0]["values"]]
        values = [float(v) for _, v in result[0]["values"]]
        data = [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, values)]

        return {
            "status": "success",
            "message": "Metrics fetched successfully",
            "query": query,
            "data": data
        }


---3—


# api.py
from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, Optional
from anomaly_detector import AnomalyDetector
import pandas as pd

app = FastAPI(
    title="Anomaly Detection and RCA API",
    description="API for detecting anomalies and analyzing cascading failures within a single namespace with persistent models.",
    version="1.0.0"
)

# Pydantic models
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

def setup_routes(detector: AnomalyDetector):
    @app.post("/train")
    async def train_model(request: QueryRequest) -> Dict[str, Any]:
        try:
            return detector.train_model(
                services=request.services,
                namespace=request.namespace,
                retrain=request.retrain,
                contamination=request.contamination
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    @app.post("/predict")
    async def predict_anomalies(request: QueryRequest, enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        try:
            df = detector.fetch_prometheus_data(lookback="1h", services=request.services, namespace=request.namespace)
            if df.empty:
                raise ValueError(f"No data fetched for the last hour in {request.namespace}")
            
            result = detector.predict_anomalies(
                df=df,
                services=request.services,
                namespace=request.namespace,
                enable_2min_duration=enable_2min_duration,
                enable_5min_duration=enable_5min_duration
            )
            return {k: v for k, v in result.items() if k != "last_hour_df"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.post("/predict_from_csv")
    async def predict_from_csv(request: PredictFromCSVRequest, file: UploadFile = File(...), enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file.file, index_col=0, parse_dates=True)
            if df.empty:
                raise ValueError("Uploaded CSV file is empty")

            result = detector.predict_anomalies(
                df=df,
                services=None,
                namespace=request.namespace,
                enable_2min_duration=enable_2min_duration,
                enable_5min_duration=enable_5min_duration
            )
            return {k: v for k, v in result.items() if k != "last_hour_df"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/graph")
    async def get_graph_image() -> Response:
        if not os.path.exists(detector.graph_image_path):
            raise HTTPException(status_code=404, detail="Graph image not found. Run /train first.")
        
        with open(detector.graph_image_path, "rb") as f:
            image_data = f.read()
        
        return Response(content=image_data, media_type="image/png")

    @app.get("/issue_graph")
    async def get_issue_graph_image() -> Response:
        if not os.path.exists(detector.issue_graph_image_path):
            raise HTTPException(status_code=404, detail="Issue graph image not found. Run /predict or /predict_from_csv first.")
        
        with open(detector.issue_graph_image_path, "rb") as f:
            image_data = f.read()
        
        return Response(content=image_data, media_type="image/png")

    @app.post("/fetch_metrics")
    async def fetch_metrics(request: FetchMetricsRequest) -> Dict[str, Any]:
        try:
            return detector.fetch_metrics(
                query=request.query,
                start_time=request.start_time,
                end_time=request.end_time,
                step=request.step
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")



————
# main.py
from anomaly_detector import AnomalyDetector
from api import app, setup_routes
import uvicorn

# Configuration
PROMETHEUS_URL = "http://prometheus:9090"
PROXY_URL = "http://proxy:3128"
PROXY_USERNAME = None
PROXY_PASSWORD = None
MODEL_DIR = "models"
GRAPH_DIR = "graphs"
DATA_DIR = "data"
PREDICTION_DIR = "predictions"
GRAPH_IMAGE_PATH = "dependency_graph.png"
ISSUE_GRAPH_IMAGE_PATH = "issue_graph.png"
QUERY_CONFIG_PATH = "config/prometheus_queries.json"
SERVICE_CONFIG_PATH = "config/service_names.json"
SERVICES = 3
DEFAULT_NAMESPACE = "ring1"
DEFAULT_CONTAMINATION = 0.005
OUTLIER_TRIM_PERCENTILE = 10

# Initialize AnomalyDetector
detector = AnomalyDetector(
    prometheus_url=PROMETHEUS_URL,
    proxy_url=PROXY_URL,
    proxy_username=PROXY_USERNAME,
    proxy_password=PROXY_PASSWORD,
    model_dir=MODEL_DIR,
    graph_dir=GRAPH_DIR,
    data_dir=DATA_DIR,
    prediction_dir=PREDICTION_DIR,
    graph_image_path=GRAPH_IMAGE_PATH,
    issue_graph_image_path=ISSUE_GRAPH_IMAGE_PATH,
    query_config_path=QUERY_CONFIG_PATH,
    service_config_path=SERVICE_CONFIG_PATH,
    services=SERVICES,
    default_namespace=DEFAULT_NAMESPACE,
    default_contamination=DEFAULT_CONTAMINATION,
    outlier_trim_percentile=OUTLIER_TRIM_PERCENTILE
)

# Setup FastAPI routes
setup_routes(detector)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
