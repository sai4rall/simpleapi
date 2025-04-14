Thank you for reporting that the /train endpoint is not working and for clarifying that the data points (features) for training the Isolation Forest model should be structured as service_name+metric_name (e.g., service_1_http_requests_total, service_1_memory_usage_pod_1, etc.). This suggests that the training data should be organized such that each feature in the model corresponds to a unique combination of service and metric (including pod indices for pod-level metrics), and the current implementation is not correctly forming these features, causing the /train endpoint to fail.
Problem Analysis
The issue with /train likely stems from how the training data is prepared in the filtered_data list for the Isolation Forest model. Currently, the code:
	•	Collects metric values into service_data for each service and metric, appending all columns (e.g., service_1_memory_usage_pod_1, service_1_memory_usage_pod_2) without explicitly mapping them to service_name+metric_name features.
	•	Combines data with np.vstack, which may misalign features or include incorrect dimensions, leading to training failures (e.g., model fitting errors or empty data).
	•	Does not ensure that each feature is uniquely identified as service_name+metric_name (or service_name+metric_name+pod_index for pod metrics).
The requirement for service_name+metric_name implies:
	•	Each column in the grouped CSV (e.g., service_1_http_requests_total, service_1_memory_usage_pod_1) should be a feature in the training data.
	•	The model should train on a matrix where columns are these features, and rows are time-aligned observations.
	•	Pod-level metrics should include pod indices in the feature name (e.g., service_1_memory_usage_pod_1).
Proposed Fix
I’ll update the program to:
	1	Restructure Training Data:
	◦	Use the grouped CSV columns (e.g., service_1_http_requests_total, service_1_memory_usage_pod_1) as features.
	◦	Align data by timestamp to create a feature matrix where each column is service_name+metric_name (or +pod_index).
	2	Fix /train:
	◦	Ensure filtered_data is built with one feature per unique service_name+metric_name+pod_index.
	◦	Handle missing data (NaNs) and ensure business-time filtering (incoming_traffic > 0) is applied correctly.
	3	Maintain CSV Grouping:
	◦	Keep the grouped metrics.csv format, as it already uses service_name+metric_name for columns.
	4	Update Other Endpoints:
	◦	Ensure /predict and /predict_on_csv align with the new feature structure for consistency.
	◦	Verify /fetch_metrics continues to produce correct CSVs.
	5	Preserve All Features:
	◦	Retain pod indexing, non-pod metric handling, outlier filtering, cascading failures, and all prior functionality.
Updated Python Program (`main.py`)
Below is the revised program, focusing on fixing /train and ensuring features are structured as service_name+metric_name.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from string import Template
from sklearn.ensemble import IsolationForest
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI app
app = FastAPI()

# Configuration
PROMETHEUS_URL = "http://your-prometheus-server:9090"  # Update with your Prometheus URL
METRICS_FILE = "metrics.json"
SERVICES_FILE = "services.json"
MODEL_DIR = "models"
IMAGE_DIR = "images"
DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
PREDICT_DATA_DIR = os.path.join(DATA_DIR, "predict")
STEP = "5m"  # Query resolution
CONTAMINATION = 0.1  # Isolation Forest contamination parameter

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(PREDICT_DATA_DIR, exist_ok=True)

def load_config(file_path):
    """Load JSON configuration file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise

def substitute_query(query, variables):
    """Substitute Mustache-like variables in the query."""
    try:
        query = query.replace('{{', '${').replace('}}', '}')
        return Template(query).substitute(**variables)
    except KeyError as e:
        logging.error(f"Missing variable in query: {e}")
        return None
    except Exception as e:
        logging.error(f"Error substituting query: {e}")
        return None

def query_prometheus(query, start_time, end_time, step=STEP):
    """Query Prometheus API directly."""
    try:
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        url = f"{PROMETHEUS_URL}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_ts,
            "end": end_ts,
            "step": step
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data['status'] != 'success' or not data['data']['result']:
            logging.warning(f"No data for query: {query}")
            return None
        return data['data']['result']
    except Exception as e:
        logging.error(f"Error querying Prometheus: {e}")
        return None

def process_prometheus_data(result, metric_name, service):
    """Convert Prometheus query result to DataFrame, handling duplicates."""
    if not result:
        return pd.DataFrame(), False, {}
    
    data = []
    pod_indices = {}
    next_pod_index = 1
    has_pod_name = False
    
    for res in result:
        metric = res['metric']
        pod_name = metric.get('pod_name', None)
        job = metric.get('job', 'unknown')
        
        if pod_name:
            has_pod_name = True
            pod_key = (pod_name, job)
            if pod_key not in pod_indices:
                pod_indices[pod_key] = f"pod_{next_pod_index}"
                next_pod_index += 1
            identifier = pod_indices[pod_key]
            col_name = f"{service}_{metric_name}_{identifier}"
        else:
            identifier = 'none'
            col_name = f"{service}_{metric_name}"
        
        for timestamp, value in res['values']:
            try:
                data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='s'),
                    'column': col_name,
                    'value': float(value)
                })
            except ValueError:
                continue
    
    df = pd.DataFrame(data)
    if df.empty:
        return df, has_pod_name, pod_indices
    
    # Aggregate duplicates
    df = df.groupby(['timestamp', 'column'])['value'].mean().reset_index()
    
    try:
        df_pivot = df.pivot(index='timestamp', columns='column', values='value')
    except ValueError as e:
        logging.error(f"Pivot failed for {metric_name} in {service}: {e}")
        df_pivot = df[['timestamp', 'value']].rename(columns={'value': col_name})
    
    df_pivot.reset_index(inplace=True)
    return df_pivot, has_pod_name, pod_indices

def save_grouped_csv(dataframes, namespace, data_dir, filename="metrics.csv"):
    """Save all dataframes into a single CSV grouped by timestamp."""
    if not dataframes:
        return None
    
    combined_df = None
    for service in dataframes:
        for metric, (df, _, _) in dataframes[service].items():
            if not df.empty:
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on='timestamp', how='outer')
    
    if combined_df is None or combined_df.empty:
        return None
    
    combined_df.sort_values('timestamp', inplace=True)
    
    os.makedirs(os.path.join(data_dir, namespace), exist_ok=True)
    csv_path = os.path.join(data_dir, namespace, filename)
    combined_df.to_csv(csv_path, index=False)
    logging.info(f"Saved grouped data to {csv_path}")
    return csv_path

def filter_outliers(df, columns):
    """Remove top/bottom 10% outliers for specified columns."""
    if df.empty:
        return df
    df_filtered = df.copy()
    for col in columns:
        if col in df.columns:
            lower_bound = df[col].quantile(0.1)
            upper_bound = df[col].quantile(0.9)
            df_filtered = df_filtered[
                (df_filtered[col].isna()) | 
                ((df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound))
            ]
    return df_filtered

def generate_service_graph(dataframes, services):
    """Generate a service dependency graph with weighted edges based on correlations."""
    G = nx.DiGraph()
    for service in services:
        G.add_node(service)
    
    for i, s1 in enumerate(services):
        for s2 in services[i+1:]:
            if s1 in dataframes and s2 in dataframes:
                for metric in dataframes[s1]:
                    if metric in dataframes[s2]:
                        try:
                            df1, _, _ = dataframes[s1][metric]
                            df2, _, _ = dataframes[s2][metric]
                            if not df1.empty and not df2.empty:
                                merged = df1.merge(df2, on='timestamp', suffixes=('_1', '_2'))
                                for col1 in [c for c in merged.columns if c.endswith('_1') or f"{s1}_{metric}" in c]:
                                    for col2 in [c for c in merged.columns if c.endswith('_2') or f"{s2}_{metric}" in c]:
                                        if col1 != 'timestamp' and col2 != 'timestamp':
                                            corr = merged[col1].corr(merged[col2])
                                            if not np.isnan(corr) and abs(corr) > 0.5:
                                                G.add_edge(s1, s2, weight=abs(corr))
                                                G.add_edge(s2, s1, weight=abs(corr))
                        except Exception as e:
                            logging.warning(f"Error computing correlation between {s1} and {s2}: {e}")
    
    return G

def save_graph_image(G, filename, title):
    """Save network graph as an image."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.title(title)
    plt.savefig(filename, format='png')
    plt.close()

class TrainRequest(BaseModel):
    namespace: str
    last_days: Optional[int] = 7

class PredictRequest(BaseModel):
    namespace: str
    last_minutes: Optional[int] = 60
    two_minutes: Optional[bool] = False
    five_minutes: Optional[bool] = False

class FetchMetricsRequest(BaseModel):
    namespace: str
    last_minutes: Optional[int] = 60

@app.post("/train")
async def train_model(request: TrainRequest):
    """Train Isolation Forest model for a namespace."""
    namespace = request.namespace
    last_days = request.last_days
    
    metrics_config = load_config(METRICS_FILE)
    services_config = load_config(SERVICES_FILE)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=last_days)
    
    dataframes = {}
    services = [s['service_name'] for s in services_config['services']]
    
    incoming_traffic = {}
    
    for service_config in services_config['services']:
        service = service_config['service_name']
        dataframes[service] = {}
        variables = {
            'namespace': namespace,
            'jobname': service_config.get('jobname', ''),
            'dbname': service_config.get('dbname', '') or ''
        }
        
        for metric in metrics_config:
            metric_name = metric['metric_name']
            query_template = metric['query']
            applicable_to = metric['applicable_to']
            
            if applicable_to != 'all' and service not in applicable_to:
                continue
            
            query = substitute_query(query_template, variables)
            if not query:
                continue
            
            result = query_prometheus(query, start_time, end_time)
            df, has_pod_name, pod_indices = process_prometheus_data(result, metric_name, service)
            
            if metric_name == 'incoming_traffic':
                incoming_traffic[service] = (df, has_pod_name, pod_indices)
            else:
                df = filter_outliers(df, [c for c in df.columns if c != 'timestamp'])
                dataframes[service][metric_name] = (df, has_pod_name, pod_indices)
    
    csv_path = save_grouped_csv(dataframes, namespace, TRAIN_DATA_DIR)
    if not csv_path:
        logging.warning("No training data saved")
    
    # Prepare training data with service_name+metric_name features
    feature_columns = []
    for service in services:
        for metric in metrics_config:
            metric_name = metric['metric_name']
            if metric_name == 'incoming_traffic':
                continue
            applicable_to = metric['applicable_to']
            if applicable_to != 'all' and service not in applicable_to:
                continue
            if metric_name in dataframes[service]:
                df, has_pod_name, pod_indices = dataframes[service][metric_name]
                if has_pod_name:
                    for pod_key, pod_index in pod_indices.items():
                        feature_columns.append(f"{service}_{metric_name}_{pod_index}")
                else:
                    feature_columns.append(f"{service}_{metric_name}")
    
    if not feature_columns:
        raise HTTPException(status_code=400, detail="No valid features for training")
    
    # Combine data into a single DataFrame
    combined_df = None
    for service in dataframes:
        for metric, (df, _, _) in dataframes[service].items():
            if metric == 'incoming_traffic' or df.empty:
                continue
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='timestamp', how='outer')
    
    if combined_df is None or combined_df.empty:
        raise HTTPException(status_code=400, detail="No valid data for training")
    
    # Filter business-time data
    traffic_timestamps = set()
    for service in incoming_traffic:
        traffic_df, _, _ = incoming_traffic[service]
        if traffic_df.empty:
            continue
        traffic_cols = [c for c in traffic_df.columns if c != 'timestamp']
        if traffic_cols:
            traffic_timestamps.update(traffic_df[traffic_df[traffic_cols].gt(0).any(axis=1)]['timestamp'])
    
    if traffic_timestamps:
        combined_df = combined_df[combined_df['timestamp'].isin(traffic_timestamps)]
    
    if combined_df.empty:
        raise HTTPException(status_code=400, detail="No business-time data for training")
    
    # Prepare feature matrix
    feature_data = combined_df[[c for c in combined_df.columns if c in feature_columns]].values
    feature_data = np.nan_to_num(feature_data, nan=0.0)
    
    if feature_data.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid feature data for training")
    
    # Train Isolation Forest
    model = IsolationForest(contamination=CONTAMINATION, random_state=42)
    model.fit(feature_data)
    
    model_path = os.path.join(MODEL_DIR, f"{namespace}_model.pkl")
    joblib.dump(model, model_path)
    
    G = generate_service_graph(dataframes, services)
    graph_path = os.path.join(MODEL_DIR, f"{namespace}_graph.pkl")
    joblib.dump(G, graph_path)
    
    image_path = os.path.join(IMAGE_DIR, f"{namespace}_service_graph.png")
    save_graph_image(G, image_path, f"Service Graph for {namespace}")
    
    return {"status": "success", "namespace": namespace, "model_path": model_path, "graph_image": image_path}

@app.post("/fetch_metrics")
async def fetch_metrics(request: FetchMetricsRequest):
    """Fetch metrics from Prometheus and save to CSV."""
    namespace = request.namespace
    last_minutes = request.last_minutes
    
    metrics_config = load_config(METRICS_FILE)
    services_config = load_config(SERVICES_FILE)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=last_minutes)
    
    dataframes = {}
    services = [s['service_name'] for s in services_config['services']]
    
    for service_config in services_config['services']:
        service = service_config['service_name']
        dataframes[service] = {}
        variables = {
            'namespace': namespace,
            'jobname': service_config.get('jobname', ''),
            'dbname': service_config.get('dbname', '') or ''
        }
        
        for metric in metrics_config:
            metric_name = metric['metric_name']
            query_template = metric['query']
            applicable_to = metric['applicable_to']
            
            if applicable_to != 'all' and service not in applicable_to:
                continue
            
            query = substitute_query(query_template, variables)
            if not query:
                continue
            
            result = query_prometheus(query, start_time, end_time)
            df, has_pod_name, pod_indices = process_prometheus_data(result, metric_name, service)
            dataframes[service][metric_name] = (df, has_pod_name, pod_indices)
    
    csv_path = save_grouped_csv(dataframes, namespace, PREDICT_DATA_DIR)
    if not csv_path:
        raise HTTPException(status_code=400, detail="No metrics data fetched")
    
    return {"status": "success", "namespace": namespace, "csv_file": csv_path}

@app.post("/predict")
async def predict_anomalies(request: PredictRequest):
    """Predict anomalies using fresh Prometheus data."""
    namespace = request.namespace
    last_minutes = request.last_minutes
    two_minutes = request.two_minutes
    five_minutes = request.five_minutes
    
    model_path = os.path.join(MODEL_DIR, f"{namespace}_model.pkl")
    graph_path = os.path.join(MODEL_DIR, f"{namespace}_graph.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail=f"Model or graph not found for {namespace}")
    
    model = joblib.load(model_path)
    G = joblib.load(graph_path)
    
    metrics_config = load_config(METRICS_FILE)
    services_config = load_config(SERVICES_FILE)
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=last_minutes)
    
    dataframes = {}
    services = [s['service_name'] for s in services_config['services']]
    
    for service_config in services_config['services']:
        service = service_config['service_name']
        dataframes[service] = {}
        variables = {
            'namespace': namespace,
            'jobname': service_config.get('jobname', ''),
            'dbname': service_config.get('dbname', '') or ''
        }
        
        for metric in metrics_config:
            metric_name = metric['metric_name']
            query_template = metric['query']
            applicable_to = metric['applicable_to']
            
            if applicable_to != 'all' and service not in applicable_to:
                continue
            
            query = substitute_query(query_template, variables)
            if not query:
                continue
            
            result = query_prometheus(query, start_time, end_time)
            df, has_pod_name, pod_indices = process_prometheus_data(result, metric_name, service)
            dataframes[service][metric_name] = (df, has_pod_name, pod_indices)
    
    csv_path = save_grouped_csv(dataframes, namespace, PREDICT_DATA_DIR)
    if not csv_path:
        logging.warning("No prediction data saved")
    
    return predict_common(namespace, dataframes, services, model, G, two_minutes, five_minutes)

@app.post("/predict_on_csv")
async def predict_on_csv(request: PredictRequest):
    """Predict anomalies using saved CSV files."""
    namespace = request.namespace
    last_minutes = request.last_minutes
    two_minutes = request.two_minutes
    five_minutes = request.five_minutes
    
    model_path = os.path.join(MODEL_DIR, f"{namespace}_model.pkl")
    graph_path = os.path.join(MODEL_DIR, f"{namespace}_graph.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail=f"Model or graph not found for {namespace}")
    
    model = joblib.load(model_path)
    G = joblib.load(graph_path)
    
    services_config = load_config(SERVICES_FILE)
    metrics_config = load_config(METRICS_FILE)
    
    services = [s['service_name'] for s in services_config['services']]
    dataframes = {}
    
    csv_path = os.path.join(PREDICT_DATA_DIR, namespace, "metrics.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"No CSV data found for namespace {namespace}")
    
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=last_minutes)
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid CSV data within time range")
        
        for service in services:
            dataframes[service] = {}
            for metric in metrics_config:
                metric_name = metric['metric_name']
                applicable_to = metric['applicable_to']
                if applicable_to != 'all' and service not in applicable_to:
                    continue
                
                metric_cols = [c for c in df.columns if c.startswith(f"{service}_{metric_name}")]
                if metric_cols:
                    metric_df = df[['timestamp'] + metric_cols].copy()
                    has_pod_name = any('_pod_' in c for c in metric_cols)
                    pod_indices = {}
                    if has_pod_name:
                        for col in metric_cols:
                            if '_pod_' in col:
                                pod_index = col.split('_pod_')[-1]
                                pod_indices[(col, metric_name)] = pod_index
                    else:
                        pod_indices[(metric_cols[0], metric_name)] = 'none'
                    dataframes[service][metric_name] = (metric_df, has_pod_name, pod_indices)
    
    except Exception as e:
        logging.error(f"Error reading CSV {csv_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {e}")
    
    if not any(dataframes[s] for s in dataframes):
        raise HTTPException(status_code=400, detail="No valid CSV data found for prediction")
    
    return predict_common(namespace, dataframes, services, model, G, two_minutes, five_minutes)

def predict_common(namespace, dataframes, services, model, G, two_minutes, five_minutes):
    """Common prediction logic for /predict and /predict_on_csv."""
    feature_columns = []
    for service in services:
        for metric_name in [m['metric_name'] for m in load_config(METRICS_FILE) if m['metric_name'] != 'incoming_traffic']:
            if metric_name in dataframes[service]:
                df, has_pod_name, pod_indices = dataframes[service][metric_name]
                if has_pod_name:
                    for pod_key, pod_index in pod_indices.items():
                        feature_columns.append(f"{service}_{metric_name}_{pod_index}")
                else:
                    feature_columns.append(f"{service}_{metric_name}")
    
    if not feature_columns:
        raise HTTPException(status_code=400, detail="No valid features for prediction")
    
    combined_df = None
    for service in dataframes:
        for metric, (df, _, _) in dataframes[service].items():
            if metric == 'incoming_traffic' or df.empty:
                continue
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='timestamp', how='outer')
    
    if combined_df is None or combined_df.empty:
        raise HTTPException(status_code=400, detail="No data for prediction")
    
    feature_data = combined_df[[c for c in combined_df.columns if c in feature_columns]].values
    feature_data = np.nan_to_num(feature_data, nan=0.0)
    
    if feature_data.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid feature data for prediction")
    
    predictions = model.predict(feature_data)
    anomaly_indices = np.where(predictions == -1)[0]
    
    anomaly_details = []
    
    timestamps = combined_df['timestamp'].values
    for idx in anomaly_indices:
        timestamp = timestamps[idx]
        # Find which feature caused the anomaly
        for col_idx, col in enumerate(feature_columns):
            if col_idx < feature_data.shape[1] and idx < len(feature_data):
                service = col.split('_')[0]
                metric_parts = col.split('_')[1:]
                pod_index = metric_parts[-1] if '_pod_' in col else 'none'
                metric_name = '_'.join(metric_parts[:-1]) if pod_index != 'none' else '_'.join(metric_parts)
                
                persistent = True
                if two_minutes or five_minutes:
                    duration = timedelta(minutes=5 if five_minutes else 2)
                    end_check = timestamp + duration
                    check_indices = np.where((timestamps >= timestamp) & (timestamps <= end_check))[0]
                    if len(check_indices) == 0:
                        persistent = False
                    else:
                        check_data = feature_data[check_indices]
                        check_preds = model.predict(check_data)
                        persistent = np.any(check_preds == -1)
                
                if persistent:
                    anomaly_details.append({
                        'service': service,
                        'pod_index': pod_index,
                        'metric': metric_name,
                        'timestamp': str(timestamp),
                        'value': feature_data[idx, col_idx] if col_idx < feature_data.shape[1] else None
                    })
    
    cascading_failures = []
    if anomaly_details:
        root_anomaly = min(anomaly_details, key=lambda x: x['timestamp'])
        root_service = root_anomaly['service']
        cascading_failures.append(root_anomaly)
        
        visited = set()
        queue = [root_service]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in G.successors(current):
                if neighbor not in visited:
                    neighbor_anomalies = [
                        a for a in anomaly_details
                        if a['service'] == neighbor and pd.to_datetime(a['timestamp']) > pd.to_datetime(root_anomaly['timestamp'])
                    ]
                    cascading_failures.extend(neighbor_anomalies)
                    queue.append(neighbor)
    
    failure_G = nx.DiGraph()
    for anomaly in cascading_failures:
        node_label = f"{anomaly['service']}_{anomaly['pod_index']}"
        failure_G.add_node(node_label, timestamp=anomaly['timestamp'])
    for i, anomaly in enumerate(cascading_failures[:-1]):
        next_anomaly = cascading_failures[i + 1]
        failure_G.add_edge(
            f"{anomaly['service']}_{anomaly['pod_index']}",
            f"{next_anomaly['service']}_{next_anomaly['pod_index']}"
        )
    
    failure_image_path = os.path.join(IMAGE_DIR, f"{namespace}_cascading_failure.png")
    save_graph_image(failure_G, failure_image_path, f"Cascading Failure Graph for {namespace}")
    
    return {
        "namespace": namespace,
        "anomalies": anomaly_details,
        "cascading_failures": cascading_failures,
        "cascading_failure_graph": failure_image_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
Key Changes
	1	Fixed /train:
	◦	Feature Definition:
	▪	Created feature_columns list with entries like service_1_http_requests_total, service_1_memory_usage_pod_1, etc., based on dataframes.
	▪	Includes pod indices for pod-level metrics and excludes incoming_traffic.
	◦	Data Preparation:
	▪	Combines all metric DataFrames into combined_df using merge on timestamp.
	▪	Filters for business-time data using incoming_traffic > 0.
	▪	Extracts feature matrix using only feature_columns, ensuring each column is service_name+metric_name+pod_index.
	◦	Model Training:
	▪	Trains Isolation Forest on feature_data, where columns match feature_columns.
	▪	Handles NaNs with np.nan_to_num to ensure valid input.
	◦	Error Handling:
	▪	Checks for empty data, invalid features, or missing business-time data, raising HTTP exceptions with clear messages.
	2	Updated predict_common:
	◦	Aligned with /train by using the same feature_columns structure.
	◦	Extracts features from combined_df matching service_name+metric_name.
	◦	Improved anomaly reporting:
	▪	Identifies the specific metric_name and pod_index for each anomaly.
	▪	Includes metric and value in anomaly details for clarity.
	◦	Adjusted persistence check to operate on time ranges, ensuring two_minutes and five_minutes flags work correctly.
	3	Preserved Functionality:
	◦	CSV Grouping: Still saves data/train/{namespace}/metrics.csv and data/predict/{namespace}/metrics.csv with columns like service_1_http_requests_total.
	◦	Pod Indexing: Maintains pod_1, pod_2, etc., in feature names.
	◦	Non-Pod Metrics: Handled correctly (e.g., service_1_http_requests_total).
	◦	Other Endpoints:
	▪	/fetch_metrics: Produces grouped CSVs with correct features.
	▪	/predict: Uses same feature structure for consistency.
	▪	/predict_on_csv: Reads grouped CSVs and reconstructs features.
	◦	Cascading Failures: Graphs and logic unchanged, using service-level dependencies.
Example Config Files
The existing metrics.json and services.json are compatible. Ensure metrics include both pod-level (e.g., memory_usage) and non-pod (e.g., http_requests_total) queries.
How to Test
	1	Prepare Environment:
	◦	Verify metrics.json and services.json.
	◦	Ensure directories: models/, images/, data/train/, data/predict/.
	◦	Update PROMETHEUS_URL.
	2	Run the Server: python main.py
	3	
	4	Test /train: curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"namespace": "namespace_1", "last_days": 7}'
	5	
	◦	Expected Response: {
	◦	    "status": "success",
	◦	    "namespace": "namespace_1",
	◦	    "model_path": "models/namespace_1_model.pkl",
	◦	    "graph_image": "images/namespace_1_service_graph.png"
	◦	}
	◦	
	◦	Check:
	▪	data/train/namespace_1/metrics.csv should have columns like service_1_http_requests_total, service_1_memory_usage_pod_1.
	▪	models/namespace_1_model.pkl should exist.
	6	Test Other Endpoints:
	◦	Fetch Metrics: curl -X POST "http://localhost:8000/fetch_metrics" -H "Content-Type: application/json" -d '{"namespace": "namespace_1", "last_minutes": 30}'
	◦	
	◦	Predict: curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"namespace": "namespace_1", "last_minutes": 30, "two_minutes": true}'
	◦	
	▪	Verify anomalies include metric and pod_index.
	◦	Predict on CSV: curl -X POST "http://localhost:8000/predict_on_csv" -H "Content-Type: application/json" -d '{"namespace": "namespace_1", "last_minutes": 30, "two_minutes": true}'
	◦	
Example Output
data/train/namespace_1/metrics.csv:
timestamp,service_1_http_requests_total,service_1_memory_usage_pod_1,service_2_cpu_usage,...
2025-04-13T10:00:00,1005.5,123456789.0,0.75,...
2025-04-13T10:05:00,1002.3,123500000.0,0.80,...
/predict Response:
{
    "namespace": "namespace_1",
    "anomalies": [
        {
            "service": "service_1",
            "pod_index": "none",
            "metric": "http_requests_total",
            "timestamp": "2025-04-13T10:00:00",
            "value": 1005.5
        },
        {
            "service": "service_1",
            "pod_index": "pod_1",
            "metric": "memory_usage",
            "timestamp": "2025-04-13T10:00:00",
            "value": 123456789.0
        }
    ],
    "cascading_failures": [...],
    "cascading_failure_graph": "images/namespace_1_cascading_failure.png"
}
Notes
	•	Feature Clarity: Each feature is now explicitly service_name+metric_name+pod_index, matching the CSV columns.
	•	Robustness: Checks ensure non-empty data and valid features before training.
	•	Performance: Merging DataFrames may be memory-intensive for many services; consider limiting last_days if issues arise.
	•	Debugging: If /train still fails, please share the error message or logs (e.g., stack trace, empty data warnings) for precise diagnosis.
If you have specific details about the failure (e.g., error message) or want tweaks (e.g., different feature naming), let me know, and I’ll refine further!
