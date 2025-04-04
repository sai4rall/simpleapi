from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
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
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import pystache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anomaly Detection and RCA API",
    description="API for detecting anomalies and analyzing cascading failures within a single namespace with persistent models.",
    version="1.0.0"
)

# Configuration
PROMETHEUS_URL = "http://prometheus:9090"
PROXY_URL = "http://proxy:3128"
PROXY_USERNAME = None
PROXY_PASSWORD = None
MODEL_DIR = "models"
GRAPH_DIR = "graphs"
DATA_DIR = "data"
GRAPH_IMAGE_PATH = "dependency_graph.png"
QUERY_CONFIG_PATH = "config/prometheus_queries.json"
SERVICE_CONFIG_PATH = "config/service_names.json"
SERVICES = 3  # nsm, app1, app2
DEFAULT_NAMESPACE = "ring1"
DEFAULT_CONTAMINATION = 0.005

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

proxy_dict = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None
if PROXY_USERNAME and PROXY_PASSWORD:
    proxy_dict["http"] = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_URL.split('://')[1]}"
    proxy_dict["https"] = f"https://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_URL.split('://')[1]}"

client = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True, requests_params={"proxies": proxy_dict})

with open(QUERY_CONFIG_PATH, "r") as f:
    config = json.load(f)
    METRICS = config["metrics"]

with open(SERVICE_CONFIG_PATH, "r") as f:
    service_config = json.load(f)
    DEFAULT_SERVICES = service_config["services"]
    if len(DEFAULT_SERVICES) != SERVICES:
        raise ValueError(f"Expected {SERVICES} services in {SERVICE_CONFIG_PATH}, got {len(DEFAULT_SERVICES)}")

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

def get_model_path(namespace: str) -> str:
    return os.path.join(MODEL_DIR, f"models_{namespace}.pkl")

def get_graph_path(namespace: str) -> str:
    return os.path.join(GRAPH_DIR, f"graph_{namespace}.pkl")

def get_data_path(namespace: str) -> str:
    return os.path.join(DATA_DIR, f"trained_data_{namespace}.csv")

def fetch_prometheus_data(lookback: str, services: Dict[str, Dict[str, str]], namespace: str) -> pd.DataFrame:
    """Fetch Prometheus data for a single namespace with service-specific metrics."""
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
    for service_name, props in list(services.items())[:SERVICES]:
        context = {"service_name": service_name, "namespace": namespace, **props}
        for metric in METRICS:
            applicable_services = metric.get("services", None)
            if applicable_services and service_name not in applicable_services:
                continue
            query = pystache.render(metric["query"], context)
            try:
                result = client.custom_query_range(
                    query=query,
                    start_time=start_time,
                    end_time=end_time,
                    step=step
                )
                logger.info(f"Query for {namespace}/{service_name}_{metric['name']}: {query}, Result: {len(result[0]['values']) if result else 0} values")
                if result and result[0]["values"]:
                    timestamps = [datetime.fromtimestamp(float(t)) for t, _ in result[0]["values"]]
                    values = [float(v) for _, v in result[0]["values"]]
                    col_name = f"{service_name}_{metric['name']}"
                    data[col_name] = pd.Series(values, index=timestamps)
                else:
                    data[f"{service_name}_{metric['name']}"] = pd.Series([], index=[])
            except Exception as e:
                logger.error(f"Error fetching {metric['name']} for {namespace}/{service_name}: {str(e)}")
                data[f"{service_name}_{metric['name']}"] = pd.Series([], index=[])

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    logger.info(f"DataFrame shape for {namespace}: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data with stronger smoothing to reduce noise."""
    df = df.resample("5T").mean().fillna(0)
    df = df.rolling(window=3, min_periods=1).mean()
    logger.info(f"After resampling, fillna, and smoothing, DataFrame shape: {df.shape}")
    df = df.loc[~(df == 0).all(axis=1)]
    logger.info(f"After filtering zero rows, DataFrame shape: {df.shape}")
    if df.empty:
        logger.warning("DataFrame is empty after preprocessing")
        return df
    df = (df - df.mean()) / df.std()
    return df

def infer_dependency_graph(df: pd.DataFrame, service_names: List[str]) -> nx.DiGraph:
    """Infer dependency graph for a single namespace using service names."""
    G = nx.DiGraph()
    lag_steps = 3
    for service_a in service_names:
        for service_b in service_names:
            if service_a != service_b:
                metric_a = df.get(f"{service_a}_http_requests_rate", pd.Series())
                metric_b = df.get(f"{service_b}_cpu_usage", pd.Series())
                if not metric_a.empty and not metric_b.empty:
                    aligned = pd.concat([metric_a, metric_b], axis=1, join="inner")
                    if len(aligned) > lag_steps:
                        corr = correlate(aligned.iloc[:, 0], aligned.iloc[:, 1], mode="full")
                        lag_range = corr[len(aligned) - lag_steps:len(aligned)]
                        max_corr = np.max(np.abs(lag_range))
                        if max_corr > 0.7:
                            G.add_edge(service_a, service_b, weight=max_corr)
    logger.info(f"Dependency graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def draw_graph(G: nx.DiGraph, namespace: str) -> Dict[str, Any]:
    """Draw the dependency graph with service names."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.title(f"Dependency Graph of Services in {namespace}")
    plt.savefig(GRAPH_IMAGE_PATH, format="png", dpi=300)
    plt.close()
    edges = [{"source": src, "target": dst, "weight": data["weight"]} for src, dst, data in G.edges(data=True)]
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edge_list": edges}

@app.post("/train")
async def train_model(request: QueryRequest) -> Dict[str, Any]:
    """Train per-service models for a single namespace and save to namespace-specific files."""
    try:
        services = request.services if request.services is not None else DEFAULT_SERVICES
        namespace = request.namespace if request.namespace is not None else DEFAULT_NAMESPACE
        contamination = request.contamination if request.contamination is not None else DEFAULT_CONTAMINATION
        if len(services) < SERVICES:
            raise ValueError(f"Expected {SERVICES} services, got {len(services)}")
        service_names = list(services.keys())

        model_path = get_model_path(namespace)
        graph_path = get_graph_path(namespace)
        data_path = get_data_path(namespace)

        if request.retrain or not os.path.exists(model_path) or not os.path.exists(data_path):
            logger.info(f"Training: Fetching fresh data from Prometheus for {namespace}")
            df = fetch_prometheus_data(lookback="2w", services=services, namespace=namespace)
            if df.empty:
                raise ValueError(f"No data fetched from Prometheus for {namespace}")
            df = preprocess_data(df)
            if df.empty:
                raise ValueError(f"Preprocessed DataFrame is empty for {namespace}")
            df.to_csv(data_path)
            logger.info(f"Saved preprocessed data to {data_path}")

            models = {}
            for service in service_names:
                service_cols = [col for col in df.columns if f"{service}_" in col]
                if service_cols and not df[service_cols].dropna().empty:
                    try:
                        iso_forest = IsolationForest(
                            contamination=contamination,
                            n_estimators=200,
                            random_state=42
                        )
                        models[service] = iso_forest.fit(df[service_cols].dropna())
                        logger.info(f"Trained model for {service} with {len(df[service_cols].dropna())} samples in {namespace}, contamination={contamination}")
                    except ValueError as e:
                        logger.warning(f"Skipping {service} due to insufficient data: {str(e)}")
                else:
                    logger.warning(f"No valid data for {service} in {namespace}")

            G = infer_dependency_graph(df, service_names)
            graph_summary = draw_graph(G, namespace)

            with open(model_path, "wb") as f:
                pickle.dump(models, f)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
        else:
            logger.info(f"Loading existing preprocessed data from {data_path} for {namespace}")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if df.empty:
                raise ValueError(f"Loaded DataFrame from {data_path} is empty")
            G = pickle.load(open(graph_path, "rb"))
            graph_summary = draw_graph(G, namespace)

        return {
            "status": "success",
            "message": f"Model trained or loaded, graph inferred, and image saved for {namespace}",
            "data_points": len(df),
            "features": len(df.columns),
            "graph_summary": graph_summary,
            "graph_image_url": "/graph"
        }
    except Exception as e:
        logger.error(f"Training failed for {namespace}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict_anomalies(request: QueryRequest, enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
    """Detect anomalies in the specified namespace with optional duration checks."""
    try:
        services = request.services if request.services is not None else DEFAULT_SERVICES
        namespace = request.namespace if request.namespace is not None else DEFAULT_NAMESPACE
        if len(services) < SERVICES:
            raise ValueError(f"Expected {SERVICES} services, got {len(services)}")
        service_names = list(services.keys())

        model_path = get_model_path(namespace)
        graph_path = get_graph_path(namespace)

        if not os.path.exists(model_path) or not os.path.exists(graph_path):
            raise HTTPException(status_code=400, detail=f"Model not trained for {namespace}. Run /train first.")

        with open(model_path, "rb") as f:
            models = pickle.load(f)
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        df = fetch_prometheus_data(lookback="1h", services=services, namespace=namespace)
        if df.empty:
            raise ValueError(f"No data fetched for the last hour in {namespace}")
        df = df.fillna(0)
        last_hour_df = df.tail(12)  # Last 12 minutes (1m step)

        anomalies: Dict[str, Dict[str, str]] = {}
        for service in service_names:
            service_cols = [col for col in last_hour_df.columns if f"{service}_" in col]
            if service_cols and service in models:
                preds = models[service].predict(last_hour_df[service_cols])
                anomaly_indices = last_hour_df.index[preds == -1]

                if not enable_2min_duration and not enable_5min_duration:
                    # No duration check: flag any anomaly
                    if len(anomaly_indices) > 0:
                        anomalies[service] = {}
                        for col in service_cols:
                            metric_name = col.split('_', 1)[1]
                            anomaly_mask = preds == -1
                            if anomaly_mask.any():
                                first_anomaly_time = last_hour_df.index[anomaly_mask][0]
                                anomalies[service][metric_name] = str(first_anomaly_time)

                elif enable_2min_duration:
                    # 2-minute duration: check for 2 consecutive anomalies
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
                    # 5-minute duration: check for 5 consecutive anomalies
                    if len(anomaly_indices) >= 5:
                        has_consecutive = False
                        for i in range(len(preds) - 4):  # Check 5 consecutive points
                            if all(preds[i:i+5] == -1):
                                has_consecutive = True
                                break
                        if has_consecutive:
                            anomalies[service] = {}
                            for col in service_cols:
                                metric_name = col.split('_', 1)[1]
                                anomaly_mask = preds == -1
                                if anomaly_mask.any():
                                    # Report the start of the 5-minute sequence
                                    first_anomaly_time = last_hour_df.index[i] if i > 0 else last_hour_df.index[anomaly_mask][0]
                                    anomalies[service][metric_name] = str(first_anomaly_time)

        root_cause = None
        cascading_graph: Dict[str, Dict[str, Any]] = {}
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

        return {
            "status": "success",
            "anomalies": anomalies,
            "root_cause": root_cause,
            "cascading_failures_graph": cascading_graph,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction failed for {namespace}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/graph")
async def get_graph_image() -> Response:
    """Serve the dependency graph image."""
    if not os.path.exists(GRAPH_IMAGE_PATH):
        raise HTTPException(status_code=404, detail="Graph image not found. Run /train first.")
    
    with open(GRAPH_IMAGE_PATH, "rb") as f:
        image_data = f.read()
    
    return Response(content=image_data, media_type="image/png")

@app.post("/fetch_metrics")
async def fetch_metrics(request: FetchMetricsRequest) -> Dict[str, Any]:
    """Fetch Prometheus metrics using a custom query."""
    try:
        end_time = datetime.utcnow() if request.end_time is None else datetime.fromisoformat(request.end_time.replace("Z", "+00:00"))
        start_time = end_time - timedelta(hours=1) if request.start_time is None else datetime.fromisoformat(request.start_time.replace("Z", "+00:00"))

        result = client.custom_query_range(
            query=request.query,
            start_time=start_time,
            end_time=end_time,
            step=request.step
        )

        if not result or not result[0]["values"]:
            return {"status": "success", "message": "No data found", "data": []}

        timestamps = [datetime.fromtimestamp(float(t)).isoformat() for t, _ in result[0]["values"]]
        values = [float(v) for _, v in result[0]["values"]]
        data = [{"timestamp": ts, "value": val} for ts, val in zip(timestamps, values)]

        return {
            "status": "success",
            "message": "Metrics fetched successfully",
            "query": request.query,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
