Thank you for your request to refactor the FastAPI program into a class named Model with a single entry point method predict. Based on your requirements, I’ll:
	1	Create a Model Class:
	◦	Encapsulate all functionality (data fetching, training, prediction, CSV handling, graph generation) within a class named Model.
	◦	Maintain all existing features: 100 microservices, 10 metrics, pod indexing, grouped CSVs, Isolation Forest, cascading failures, etc.
	2	Single Entry Point Method predict:
	◦	Replace the multiple FastAPI endpoints (/train, /predict, /fetch_metrics, /predict_on_csv) with a single predict method.
	◦	The predict method will handle all operations (training, fetching, predicting, predicting on CSV) based on input parameters.
	◦	Parameters will include namespace, mode (to specify train, predict, fetch, or predict_on_csv), last_days, last_minutes, two_minutes, five_minutes.
	3	Preserve Functionality:
	◦	Keep the grouped CSV format (data/{train,predict}/{namespace}/metrics.csv).
	◦	Maintain features as service_name+metric_name+pod_index.
	◦	Support pod-level and non-pod metrics, business-time filtering, outlier removal, and cascading failure graphs.
	◦	Avoid prometheus-api-client, using direct HTTP requests.
Since the FastAPI structure is no longer needed, I’ll refactor the code into a standalone Python class with a command-line interface for testing. The predict method will act as the main interface, and I’ll ensure the code remains robust and maintainable.
Assumptions
	•	Modes: The predict method will support modes: train, fetch, predict, predict_on_csv.
	•	Input Parameters:
	◦	namespace: Required for all operations.
	◦	mode: Determines the action (train, fetch, predict, predict_on_csv).
	◦	last_days: For training (default: 7).
	◦	last_minutes: For fetching/predicting (default: 60).
	◦	two_minutes, five_minutes: For anomaly persistence in prediction modes (default: False).
	•	Output:
	◦	Returns a dictionary with results (e.g., model path, CSV path, anomalies, graphs).
	◦	Saves CSVs, models, and images as before.
	•	Execution: The class can be used programmatically or via a CLI for testing.
Directory Structure
project/
├── metrics.json
├── services.json
├── models/                   # Stores .pkl files
├── images/                   # Stores graph images
├── data/
│   ├── train/
│   │   ├── namespace_1/
│   │   │   ├── metrics.csv
│   ├── predict/
│   │   ├── namespace_1/
│   │   │   ├── metrics.csv
├── model.py                  # Model class implementation
Example Config Files
The metrics.json and services.json remain unchanged.
`metrics.json`
[
    {
        "metric_name": "http_requests_total",
        "query": "rate(http_requests_total{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m])",
        "applicable_to": "all"
    },
    {
        "metric_name": "db_query_latency",
        "query": "histogram_quantile(0.99, sum(rate(db_query_duration_seconds_bucket{namespace=\"{{namespace}}\", job=\"{{jobname}}\", dbname=\"{{dbname}}\"}[5m])) by (le))",
        "applicable_to": ["service_1", "service_2"]
    },
    {
        "metric_name": "cpu_usage",
        "query": "sum(rate(container_cpu_usage_seconds_total{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m]))",
        "applicable_to": "all"
    },
    {
        "metric_name": "memory_usage",
        "query": "sum by (pod_name, job)(increase(jvm_memory_used_bytes{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m]))",
        "applicable_to": "all"
    },
    {
        "metric_name": "network_bytes",
        "query": "sum(rate(container_network_receive_bytes_total{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m]))",
        "applicable_to": "all"
    },
    {
        "metric_name": "error_rate",
        "query": "rate(http_requests_total{namespace=\"{{namespace}}\", job=\"{{jobname}}\", status=~\"5..\"}[5m])",
        "applicable_to": "all"
    },
    {
        "metric_name": "request_duration",
        "query": "rate(http_request_duration_seconds_sum{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m])",
        "applicable_to": "all"
    },
    {
        "metric_name": "db_connections",
        "query": "db_connections{namespace=\"{{namespace}}\", job=\"{{jobname}}\", dbname=\"{{dbname}}\"}",
        "applicable_to": ["service_1", "service_2"]
    },
    {
        "metric_name": "queue_length",
        "query": "queue_length{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}",
        "applicable_to": "all"
    },
    {
        "metric_name": "incoming_traffic",
        "query": "sum(rate(http_requests_total{namespace=\"{{namespace}}\", job=\"{{jobname}}\"}[5m]))",
        "applicable_to": "all"
    }
]
`services.json`
{
    "services": [
        {
            "service_name": "service_1",
            "jobname": "webapp",
            "dbname": "mydb1"
        },
        {
            "service_name": "service_2",
            "jobname": "api",
            "dbname": "mydb2"
        },
        {
            "service_name": "service_100",
            "jobname": "worker",
            "dbname": null
        }
    ]
}
Updated Python Program (`model.py`)
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
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Model:
    def __init__(self, prometheus_url="http://your-prometheus-server:9090"):
        """Initialize Model with configuration."""
        self.prometheus_url = prometheus_url
        self.metrics_file = "metrics.json"
        self.services_file = "services.json"
        self.model_dir = "models"
        self.image_dir = "images"
        self.data_dir = "data"
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.predict_data_dir = os.path.join(self.data_dir, "predict")
        self.step = "5m"
        self.contamination = 0.1
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.train_data_dir, exist_ok=True)
        os.makedirs(self.predict_data_dir, exist_ok=True)
        
        # Load configurations
        self.metrics_config = self.load_config(self.metrics_file)
        self.services_config = self.load_config(self.services_file)
        self.services = [s['service_name'] for s in self.services_config['services']]

    def load_config(self, file_path):
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise

    def substitute_query(self, query, variables):
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

    def query_prometheus(self, query, start_time, end_time, step=None):
        """Query Prometheus API directly."""
        if step is None:
            step = self.step
        try:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            url = f"{self.prometheus_url}/api/v1/query_range"
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

    def process_prometheus_data(self, result, metric_name, service):
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
        
        df = df.groupby(['timestamp', 'column'])['value'].mean().reset_index()
        
        try:
            df_pivot = df.pivot(index='timestamp', columns='column', values='value')
        except ValueError as e:
            logging.error(f"Pivot failed for {metric_name} in {service}: {e}")
            df_pivot = df[['timestamp', 'value']].rename(columns={'value': col_name})
        
        df_pivot.reset_index(inplace=True)
        return df_pivot, has_pod_name, pod_indices

    def save_grouped_csv(self, dataframes, namespace, data_dir, filename="metrics.csv"):
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

    def filter_outliers(self, df, columns):
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

    def generate_service_graph(self, dataframes, services):
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

    def save_graph_image(self, G, filename, title):
        """Save network graph as an image."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(title)
        plt.savefig(filename, format='png')
        plt.close()

    def fetch_metrics(self, namespace, start_time, end_time):
        """Fetch metrics and return dataframes."""
        dataframes = {}
        
        for service_config in self.services_config['services']:
            service = service_config['service_name']
            dataframes[service] = {}
            variables = {
                'namespace': namespace,
                'jobname': service_config.get('jobname', ''),
                'dbname': service_config.get('dbname', '') or ''
            }
            
            for metric in self.metrics_config:
                metric_name = metric['metric_name']
                query_template = metric['query']
                applicable_to = metric['applicable_to']
                
                if applicable_to != 'all' and service not in applicable_to:
                    continue
                
                query = self.substitute_query(query_template, variables)
                if not query:
                    continue
                
                result = self.query_prometheus(query, start_time, end_time)
                df, has_pod_name, pod_indices = self.process_prometheus_data(result, metric_name, service)
                dataframes[service][metric_name] = (df, has_pod_name, pod_indices)
        
        return dataframes

    def predict(self, namespace, mode, last_days=7, last_minutes=60, two_minutes=False, five_minutes=False):
        """Single entry point for training, fetching, and predicting anomalies."""
        if mode not in ['train', 'fetch', 'predict', 'predict_on_csv']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'fetch', 'predict', or 'predict_on_csv'.")
        
        if mode == 'train':
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=last_days)
            
            dataframes = self.fetch_metrics(namespace, start_time, end_time)
            
            csv_path = self.save_grouped_csv(dataframes, namespace, self.train_data_dir)
            if not csv_path:
                logging.warning("No training data saved")
            
            # Prepare features as service_name+metric_name
            feature_columns = []
            for service in self.services:
                for metric in self.metrics_config:
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
                raise ValueError("No valid features for training")
            
            combined_df = None
            incoming_traffic = {}
            for service in dataframes:
                for metric, (df, has_pod_name, pod_indices) in dataframes[service].items():
                    if metric == 'incoming_traffic':
                        incoming_traffic[service] = (df, has_pod_name, pod_indices)
                    elif not df.empty:
                        df = self.filter_outliers(df, [c for c in df.columns if c != 'timestamp'])
                        if combined_df is None:
                            combined_df = df
                        else:
                            combined_df = combined_df.merge(df, on='timestamp', how='outer')
            
            if combined_df is None or combined_df.empty:
                raise ValueError("No valid data for training")
            
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
                raise ValueError("No business-time data for training")
            
            feature_data = combined_df[[c for c in combined_df.columns if c in feature_columns]].values
            feature_data = np.nan_to_num(feature_data, nan=0.0)
            
            if feature_data.shape[1] == 0:
                raise ValueError("No valid feature data for training")
            
            model = IsolationForest(contamination=self.contamination, random_state=42)
            model.fit(feature_data)
            
            model_path = os.path.join(self.model_dir, f"{namespace}_model.pkl")
            joblib.dump(model, model_path)
            
            G = self.generate_service_graph(dataframes, self.services)
            graph_path = os.path.join(self.model_dir, f"{namespace}_graph.pkl")
            joblib.dump(G, graph_path)
            
            image_path = os.path.join(self.image_dir, f"{namespace}_service_graph.png")
            self.save_graph_image(G, image_path, f"Service Graph for {namespace}")
            
            return {
                "status": "success",
                "namespace": namespace,
                "model_path": model_path,
                "graph_image": image_path
            }
        
        elif mode == 'fetch':
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=last_minutes)
            
            dataframes = self.fetch_metrics(namespace, start_time, end_time)
            
            csv_path = self.save_grouped_csv(dataframes, namespace, self.predict_data_dir)
            if not csv_path:
                raise ValueError("No metrics data fetched")
            
            return {
                "status": "success",
                "namespace": namespace,
                "csv_file": csv_path
            }
        
        elif mode in ['predict', 'predict_on_csv']:
            model_path = os.path.join(self.model_dir, f"{namespace}_model.pkl")
            graph_path = os.path.join(self.model_dir, f"{namespace}_graph.pkl")
            
            if not os.path.exists(model_path) or not os.path.exists(graph_path):
                raise ValueError(f"Model or graph not found for {namespace}")
            
            model = joblib.load(model_path)
            G = joblib.load(graph_path)
            
            if mode == 'predict':
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=last_minutes)
                dataframes = self.fetch_metrics(namespace, start_time, end_time)
                csv_path = self.save_grouped_csv(dataframes, namespace, self.predict_data_dir)
                if not csv_path:
                    logging.warning("No prediction data saved")
            else:
                dataframes = {}
                csv_path = os.path.join(self.predict_data_dir, namespace, "metrics.csv")
                if not os.path.exists(csv_path):
                    raise ValueError(f"No CSV data found for {namespace}")
                
                try:
                    df = pd.read_csv(csv_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(minutes=last_minutes)
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                    if df.empty:
                        raise ValueError("No valid CSV data within time range")
                    
                    for service in self.services:
                        dataframes[service] = {}
                        for metric in self.metrics_config:
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
                    raise ValueError(f"Error processing CSV: {e}")
                
                if not any(dataframes[s] for s in dataframes):
                    raise ValueError("No valid CSV data for prediction")
            
            feature_columns = []
            for service in self.services:
                for metric_name in [m['metric_name'] for m in self.metrics_config if m['metric_name'] != 'incoming_traffic']:
                    if metric_name in dataframes[service]:
                        df, has_pod_name, pod_indices = dataframes[service][metric_name]
                        if has_pod_name:
                            for pod_key, pod_index in pod_indices.items():
                                feature_columns.append(f"{service}_{metric_name}_{pod_index}")
                        else:
                            feature_columns.append(f"{service}_{metric_name}")
            
            if not feature_columns:
                raise ValueError("No valid features for prediction")
            
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
                raise ValueError("No data for prediction")
            
            feature_data = combined_df[[c for c in combined_df.columns if c in feature_columns]].values
            feature_data = np.nan_to_num(feature_data, nan=0.0)
            
            if feature_data.shape[1] == 0:
                raise ValueError("No valid feature data for prediction")
            
            predictions = model.predict(feature_data)
            anomaly_indices = np.where(predictions == -1)[0]
            
            anomaly_details = []
            timestamps = combined_df['timestamp'].values
            
            for idx in anomaly_indices:
                timestamp = timestamps[idx]
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
            
            failure_image_path = os.path.join(self.image_dir, f"{namespace}_cascading_failure.png")
            self.save_graph_image(failure_G, failure_image_path, f"Cascading Failure Graph for {namespace}")
            
            return {
                "namespace": namespace,
                "anomalies": anomaly_details,
                "cascading_failures": cascading_failures,
                "cascading_failure_graph": failure_image_path
            }

def main():
    """Command-line interface for testing the Model class."""
    parser = argparse.ArgumentParser(description="Run Model operations")
    parser.add_argument('--namespace', required=True, help="Namespace to process")
    parser.add_argument('--mode', required=True, choices=['train', 'fetch', 'predict', 'predict_on_csv'], help="Operation mode")
    parser.add_argument('--last_days', type=int, default=7, help="Days for training data")
    parser.add_argument('--last_minutes', type=int, default=60, help="Minutes for fetching/predicting")
    parser.add_argument('--two_minutes', action='store_true', help="Require anomalies to persist for 2 minutes")
    parser.add_argument('--five_minutes', action='store_true', help="Require anomalies to persist for 5 minutes")
    
    args = parser.parse_args()
    
    model = Model()
    try:
        result = model.predict(
            namespace=args.namespace,
            mode=args.mode,
            last_days=args.last_days,
            last_minutes=args.last_minutes,
            two_minutes=args.two_minutes,
            five_minutes=args.five_minutes
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
Explanation of Changes
	1	Class Structure:
	◦	Created Model class with all logic encapsulated.
	◦	Initialized with configuration (Prometheus URL, file paths, directories).
	◦	Moved helper methods (load_config, substitute_query, etc.) into the class.
	2	Single Entry Point: predict:
	◦	Handles four modes:
	▪	train: Fetches data for last_days, filters business-time data, trains Isolation Forest, saves model and graph.
	▪	fetch: Fetches data for last_minutes, saves grouped CSV.
	▪	predict: Fetches data, loads model, predicts anomalies, generates cascading failures.
	▪	predict_on_csv: Reads grouped CSV, predicts anomalies using saved model.
	◦	Parameters:
	▪	namespace: Required for all modes.
	▪	mode: Specifies operation (train, fetch, predict, predict_on_csv).
	▪	last_days, last_minutes, two_minutes, five_minutes: Control data range and anomaly persistence.
	◦	Returns a dictionary with operation-specific results (e.g., model path, CSV path, anomalies).
	3	Preserved Functionality:
	◦	Features: Uses service_name+metric_name+pod_index (e.g., service_1_memory_usage_pod_1) as features.
	◦	CSVs: Saves grouped metrics.csv in data/{train,predict}/{namespace}/.
	◦	Pod Indexing: Maintains pod_1, pod_2, etc., for pod-level metrics.
	◦	Non-Pod Metrics: Handles correctly (e.g., service_1_http_requests_total).
	◦	Business-Time Filtering: Ensures incoming_traffic > 0.
	◦	Outlier Removal: Filters top/bottom 10% for training.
	◦	Cascading Failures: Generates graphs based on earliest anomaly.
	4	CLI for Testing:
	◦	Added main() with argparse to test the class via command line.
	◦	Example commands below.
How to Run
	1	Prepare Environment:
	◦	Ensure metrics.json and services.json are in the project root.
	◦	Verify directories: models/, images/, data/train/, data/predict/.
	◦	Update PROMETHEUS_URL in the code.
	2	Run Commands:
	◦	Train: python model.py --namespace namespace_1 --mode train --last_days 7
	◦	 Output: {
	◦	  "status": "success",
	◦	  "namespace": "namespace_1",
	◦	  "model_path": "models/namespace_1_model.pkl",
	◦	  "graph_image": "images/namespace_1_service_graph.png"
	◦	}
	◦	
	◦	Fetch Metrics: python model.py --namespace namespace_1 --mode fetch --last_minutes 30
	◦	 Output: {
	◦	  "status": "success",
	◦	  "namespace": "namespace_1",
	◦	  "csv_file": "data/predict/namespace_1/metrics.csv"
	◦	}
	◦	
	◦	Predict: python model.py --namespace namespace_1 --mode predict --last_minutes 30 --two_minutes
	◦	 Output: {
	◦	  "namespace": "namespace_1",
	◦	  "anomalies": [
	◦	    {
	◦	      "service": "service_1",
	◦	      "pod_index": "none",
	◦	      "metric": "http_requests_total",
	◦	      "timestamp": "2025-04-13T10:00:00",
	◦	      "value": 1005.5
	◦	    },
	◦	    ...
	◦	  ],
	◦	  "cascading_failures": [...],
	◦	  "cascading_failure_graph": "images/namespace_1_cascading_failure.png"
	◦	}
	◦	
	◦	Predict on CSV: python model.py --namespace namespace_1 --mode predict_on_csv --last_minutes 30 --two_minutes
	◦	
	3	Programmatic Use: from model import Model
	4	model = Model()
	5	result = model.predict(namespace="namespace_1", mode="train", last_days=7)
	6	print(result)
	7	
Notes
	•	Single Entry Point: The predict method consolidates all operations, reducing complexity.
	•	Error Handling: Raises ValueError for invalid inputs or data issues, with detailed logging.
	•	Performance: Merging large DataFrames may be memory-intensive; adjust last_days or last_minutes for large systems.
	•	Extensibility: The class structure makes it easy to add new modes or features.
	•	Debugging: If issues persist, check logs for specific errors (e.g., empty data, feature mismatches).
If you need further refinements (e.g., specific error details, additional parameters, or a different interface), please let me know, and I’ll adjust the code accordingly!
