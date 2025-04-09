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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
        # Configuration
        self.prometheus_url = "http://prometheus:9090"
        self.proxy_url = "http://proxy:3128"
        self.proxy_username = None
        self.proxy_password = None
        self.model_dir = "models"
        self.graph_dir = "graphs"
        self.data_dir = "data"
        self.prediction_dir = "predictions"
        self.graph_image_path = "dependency_graph.png"
        self.issue_graph_image_path = "issue_graph.png"
        self.query_config_path = "config/prometheus_queries.json"
        self.service_config_path = "config/service_names.json"
        self.services = 3
        self.default_namespace = "ring1"
        self.default_contamination = 0.005
        self.outlier_trim_percentile = 10

        # Initialize directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Proxy setup
        self.proxy_dict = {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
        if self.proxy_username and self.proxy_password:
            self.proxy_dict["http"] = f"http://{self.proxy_username}:{self.proxy_password}@{self.proxy_url.split('://')[1]}"
            self.proxy_dict["https"] = f"https://{self.proxy_username}:{self.proxy_password}@{self.proxy_url.split('://')[1]}"

        # Prometheus client
        self.client = PrometheusConnect(url=self.prometheus_url, disable_ssl=True, requests_params={"proxies": self.proxy_dict})

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
        """Fetch Prometheus data, aggregating by job and including pod metrics as features."""
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
                query = f"sum by (job) ({pystache.render(metric['query'], context)})"
                try:
                    result = self.client.custom_query_range(
                        query=query,
                        start_time=start_time,
                        end_time=end_time,
                        step=step
                    )
                    logger.info(f"Query for {namespace}/{service_name}_{metric['name']}: {query}, Result: {len(result)} series")
                    
                    if not result:
                        logger.warning(f"No data returned for query: {query}")
                        continue

                    for series in result:
                        labels = series.get("metric", {})
                        labels.pop("__name__", None)
                        job = labels.get("job", "unknown_job")
                        col_name = f"{job}_{metric['name']}"

                        if series["values"]:
                            timestamps = [datetime.fromtimestamp(float(t)) for t, _ in series["values"]]
                            values = [float(v) for _, v in series["values"]]
                            data[col_name] = pd.Series(values, index=timestamps)
                        else:
                            data[col_name] = pd.Series([], index=[])

                    pod_query = pystache.render(metric["query"], context)
                    pod_result = self.client.custom_query_range(
                        query=pod_query,
                        start_time=start_time,
                        end_time=end_time,
                        step=step
                    )
                    for series in pod_result:
                        labels = series.get("metric", {})
                        job = labels.get("job", "unknown_job")
                        pod = labels.get("pod", "unknown_pod")
                        col_name = f"{job}_{pod}_{metric['name']}"
                        if series["values"]:
                            timestamps = [datetime.fromtimestamp(float(t)) for t, _ in series["values"]]
                            values = [float(v) for _, v in series["values"]]
                            data[col_name] = pd.Series(values, index=timestamps)
                        else:
                            data[col_name] = pd.Series([], index=[])

                except Exception as e:
                    logger.error(f"Error fetching {metric['name']} for {namespace}/{service_name}: {str(e)}")
                    data[f"{service_name}_{metric['name']}"] = pd.Series([], index=[])

        df = pd.DataFrame(data)
        if df.empty:
            logger.warning(f"DataFrame is empty for {namespace}")
            return df

        df.index = pd.to_datetime(df.index)
        logger.info(f"DataFrame shape for {namespace}: {df.shape}")
        return df

    def remove_outliers(self, df: pd.DataFrame, percentile: float = OUTLIER_TRIM_PERCENTILE) -> pd.DataFrame:
        """Remove the top and bottom percentile of values for each column in the DataFrame."""
        if df.empty:
            logger.warning("DataFrame is empty, skipping outlier removal")
            return df

        logger.info(f"Removing top {percentile}% and bottom {percentile}% outliers from DataFrame")
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
        logger.info(f"After removing outliers, DataFrame shape: {df_cleaned.shape}")
        return df_cleaned

    def preprocess_data(self, df: pd.DataFrame, remove_outliers_flag: bool = False) -> pd.DataFrame:
        """Preprocess data with smoothing and optional outlier removal."""
        if df.empty:
            logger.warning("DataFrame is empty before preprocessing")
            return df

        df = df.resample("5T").mean().fillna(0)
        df = df.rolling(window=3, min_periods=1).mean()
        logger.info(f"After resampling, fillna, and smoothing, DataFrame shape: {df.shape}")
        
        df = df.loc[~(df == 0).all(axis=1)]
        logger.info(f"After filtering zero rows, DataFrame shape: {df.shape}")
        
        if df.empty:
            logger.warning("DataFrame is empty after preprocessing")
            return df
        
        if remove_outliers_flag:
            df = self.remove_outliers(df)
            if df.empty:
                logger.warning("DataFrame is empty after outlier removal")
                return df
        
        df = (df - df.mean()) / df.std()
        return df

    def infer_dependency_graph(self, df: pd.DataFrame, job_names: List[str]) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
        """Infer dependency graph between jobs."""
        G = nx.DiGraph()
        lag_steps = 3
        job_columns = {job: [] for job in job_names}

        for col in df.columns:
            for job in job_names:
                if col.startswith(f"{job}_"):
                    job_columns[job].append(col)
                    break

        for job_a in job_names:
            for job_b in job_names:
                if job_a != job_b:
                    for metric_a in job_columns[job_a]:
                        for metric_b in job_columns[job_b]:
                            series_a = df.get(metric_a, pd.Series())
                            series_b = df.get(metric_b, pd.Series())
                            if not series_a.empty and not series_b.empty:
                                aligned = pd.concat([series_a, series_b], axis=1, join="inner")
                                if len(aligned) > lag_steps:
                                    corr = correlate(aligned.iloc[:, 0], aligned.iloc[:, 1], mode="full")
                                    lag_range = corr[len(aligned) - lag_steps:len(aligned)]
                                    max_corr = np.max(np.abs(lag_range))
                                    if max_corr > 0.7:
                                        G.add_edge(job_a, job_b, weight=max_corr)
                                        break
        logger.info(f"Dependency graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G, job_columns

    def draw_graph(self, G: nx.DiGraph, namespace: str) -> Dict[str, Any]:
        """Draw the dependency graph with job names."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(f"Dependency Graph of Jobs in {namespace}")
        plt.savefig(self.graph_image_path, format="png", dpi=300)
        plt.close()
        edges = [{"source": src, "target": dst, "weight": data["weight"]} for src, dst, data in G.edges(data=True)]
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edge_list": edges}

    def draw_issue_graph(self, G: nx.DiGraph, namespace: str, root_cause: Optional[str], affected_jobs: List[str]) -> Dict[str, Any]:
        """Draw the dependency graph highlighting the root cause (red) and affected jobs (yellow)."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        node_colors = []
        for node in G.nodes():
            if node == root_cause:
                node_colors.append("red")
            elif node in affected_jobs:
                node_colors.append("yellow")
            else:
                node_colors.append("lightblue")
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, arrows=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        plt.title(f"Issue Graph of Jobs in {namespace} (Red: Root Cause, Yellow: Affected)")
        plt.savefig(self.issue_graph_image_path, format="png", dpi=300)
        plt.close()
        edges = [{"source": src, "target": dst, "weight": data["weight"]} for src, dst, data in G.edges(data=True)]
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "edge_list": edges}

    def calculate_expected_and_deviation(self, df_train: pd.DataFrame, df_predict: pd.DataFrame, metric_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate expected behavior and deviation for a metric after prediction."""
        expected = df_train[metric_col].rolling(window=12, min_periods=1, center=True).median()
        expected_aligned = expected.reindex(df_predict.index, method="nearest").fillna(method="ffill").fillna(method="bfill")
        actual = df_predict[metric_col]
        deviation = (actual - expected_aligned).abs()
        return actual, expected_aligned, deviation

    def plot_metric_behavior(self, actual: pd.Series, expected: pd.Series, deviation: pd.Series, metric_name: str, namespace: str) -> str:
        """Plot actual, expected, and deviation for a metric and save as an image."""
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label="Actual", color="blue")
        plt.plot(expected.index, expected, label="Expected (Rolling Median)", color="green", linestyle="--")
        plt.plot(deviation.index, deviation, label="Deviation", color="red", linestyle="-.")
        plt.title(f"Metric Behavior: {metric_name} in {namespace}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        safe_metric_name = "".join(c if c.isalnum() or c == "_" else "_" for c in metric_name)
        image_path = f"{self.prediction_dir}/metric_behavior_{safe_metric_name}_{namespace}.png"
        plt.savefig(image_path, format="png", dpi=300)
        plt.close()
        return image_path

    def train_model(self, services: Dict[str, Dict[str, str]], namespace: str, retrain: bool, contamination: float) -> Dict[str, Any]:
        """Train per-job models, handling dynamic pods as features."""
        if len(services) < self.services:
            raise ValueError(f"Expected {self.services} services, got {len(services)}")

        model_path = self.get_model_path(namespace)
        graph_path = self.get_graph_path(namespace)
        data_path = self.get_data_path(namespace)

        if retrain or not os.path.exists(model_path) or not os.path.exists(data_path):
            logger.info(f"Training: Fetching fresh data from Prometheus for {namespace}")
            df = self.fetch_prometheus_data(lookback="2w", services=services, namespace=namespace)
            if df.empty:
                raise ValueError(f"No data fetched from Prometheus for {namespace}")
            
            df = self.preprocess_data(df, remove_outliers_flag=True)
            if df.empty:
                raise ValueError(f"Preprocessed DataFrame is empty for {namespace} after outlier removal")
            
            df.to_csv(data_path)
            logger.info(f"Saved preprocessed data to {data_path}")

            job_names = sorted(set(col.split('_')[0] for col in df.columns))
            models = {}
            G, job_columns = self.infer_dependency_graph(df, job_names)
            for job in job_names:
                job_cols = job_columns.get(job, [])
                if job_cols and not df[job_cols].dropna().empty:
                    try:
                        iso_forest = IsolationForest(
                            contamination=contamination,
                            n_estimators=200,
                            random_state=42
                        )
                        models[job] = iso_forest.fit(df[job_cols].dropna())
                        logger.info(f"Trained model for {job} with {len(df[job_cols].dropna())} samples in {namespace}")
                    except ValueError as e:
                        logger.warning(f"Skipping {job} due to insufficient data: {str(e)}")
                else:
                    logger.warning(f"No valid data for {job} in {namespace}")

            graph_summary = self.draw_graph(G, namespace)

            with open(model_path, "wb") as f:
                pickle.dump(models, f)
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)
        else:
            logger.info(f"Loading existing preprocessed data from {data_path} for {namespace}")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if df.empty:
                raise ValueError(f"Loaded DataFrame from {data_path} is empty")
            job_names = sorted(set(col.split('_')[0] for col in df.columns))
            G, job_columns = self.infer_dependency_graph(df, job_names)
            graph_summary = self.draw_graph(G, namespace)

        return {
            "status": "success",
            "message": f"Model trained or loaded, graph inferred, and image saved for {namespace}",
            "data_points": len(df),
            "features": len(df.columns),
            "graph_summary": graph_summary,
            "graph_image_url": "/graph"
        }

    def predict_anomalies(self, services: Dict[str, Dict[str, str]], namespace: str, enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        """Detect anomalies and generate expected behavior/deviation graphs after prediction."""
        if len(services) < self.services:
            raise ValueError(f"Expected {self.services} services, got {len(services)}")

        model_path = self.get_model_path(namespace)
        graph_path = self.get_graph_path(namespace)
        data_path = self.get_data_path(namespace)

        if not os.path.exists(model_path) or not os.path.exists(graph_path):
            raise ValueError(f"Model not trained for {namespace}. Run /train first.")

        with open(model_path, "rb") as f:
            models = pickle.load(f)
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        df_train = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if df_train.empty:
            raise ValueError(f"Training data at {data_path} is empty")

        df_predict = self.fetch_prometheus_data(lookback="1h", services=services, namespace=namespace)
        if df_predict.empty:
            raise ValueError(f"No data fetched for the last hour in {namespace}")
        
        df_predict = self.preprocess_data(df_predict, remove_outliers_flag=False)
        if df_predict.empty:
            raise ValueError(f"Preprocessed DataFrame is empty for {namespace}")
        
        df_predict = df_predict.fillna(0)
        last_hour_df = df_predict.tail(12)

        job_names = sorted(set(col.split('_')[0] for col in df_predict.columns))
        _, job_columns = self.infer_dependency_graph(df_predict, job_names)

        anomalies: Dict[str, Dict[str, str]] = {}
        anomaly_predictions = {}
        for job in job_names:
            job_cols = job_columns.get(job, [])
            if job_cols and job in models:
                trained_cols = [col for col in job_cols if col in models[job].feature_names_in_]
                if not trained_cols:
                    logger.warning(f"No matching features for {job} in trained model")
                    continue
                preds = models[job].predict(last_hour_df[trained_cols])
                anomaly_indices = last_hour_df.index[preds == -1]

                for col in trained_cols:
                    anomaly_predictions[f"{col}_anomaly"] = preds

                if not enable_2min_duration and not enable_5min_duration:
                    if len(anomaly_indices) > 0:
                        anomalies[job] = {}
                        for col in trained_cols:
                            metric_name = '_'.join(col.split('_')[1:])
                            anomaly_mask = preds == -1
                            if anomaly_mask.any():
                                first_anomaly_time = last_hour_df.index[anomaly_mask][0]
                                anomalies[job][metric_name] = str(first_anomaly_time)

        root_cause = None
        cascading_graph: Dict[str, Dict[str, Any]] = {}
        affected_jobs = []
        if anomalies:
            anomaly_times = {job: min(pd.to_datetime(list(times.values()))) for job, times in anomalies.items()}
            sorted_anomalies = sorted(anomaly_times.items(), key=lambda x: x[1])
            
            for job, time in sorted_anomalies:
                preds = list(G.predecessors(job))
                has_earlier_pred = any(pred in anomalies and anomaly_times[pred] < time for pred in preds)
                if not has_earlier_pred:
                    root_cause = job
                    break
            
            if root_cause:
                cascading_graph[root_cause] = {
                    "first_anomaly": str(anomaly_times[root_cause]),
                    "metrics": anomalies[root_cause],
                    "affected_jobs": []
                }
                visited = set([root_cause])
                queue = [root_cause]
                while queue:
                    current = queue.pop(0)
                    for succ in G.successors(current):
                        if succ in anomalies and succ not in visited:
                            cascading_graph[current]["affected_jobs"].append({
                                "job": succ,
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ]
                            })
                            cascading_graph[succ] = {
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ],
                                "affected_jobs": []
                            }
                            visited.add(succ)
                            queue.append(succ)
                            affected_jobs.append(succ)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prediction_path = self.get_prediction_path(namespace, timestamp)
        prediction_df = last_hour_df.copy()
        for col, pred_values in anomaly_predictions.items():
            prediction_df[col] = pred_values
        prediction_df.to_csv(prediction_path)
        logger.info(f"Saved prediction data to {prediction_path}")

        metric_graphs = {}
        if anomalies:
            for job, metric_times in anomalies.items():
                for metric_name, _ in metric_times.items():
                    full_metric_name = f"{job}_{metric_name}"
                    if full_metric_name in df_predict.columns and full_metric_name in df_train.columns:
                        actual, expected, deviation = self.calculate_expected_and_deviation(df_train, last_hour_df, full_metric_name)
                        graph_path = self.plot_metric_behavior(actual, expected, deviation, full_metric_name, namespace)
                        metric_graphs[full_metric_name] = f"/metric_graph?metric={full_metric_name}"

        issue_graph_summary = None
        if root_cause:
            issue_graph_summary = self.draw_issue_graph(G, namespace, root_cause, affected_jobs)

        return {
            "status": "success",
            "anomalies": anomalies,
            "root_cause": root_cause,
            "cascading_failures_graph": cascading_graph,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_file": prediction_path,
            "issue_graph_image_url": "/issue_graph" if issue_graph_summary else None,
            "metric_graphs": metric_graphs
        }

    def predict_from_csv(self, namespace: str, csv_file: Any, enable_2min_duration: bool = False, enable_5min_duration: bool = False) -> Dict[str, Any]:
        """Predict anomalies using data from an uploaded CSV file."""
        model_path = self.get_model_path(namespace)
        graph_path = self.get_graph_path(namespace)
        job_names = list(self.default_services.keys())

        if not os.path.exists(model_path) or not os.path.exists(graph_path):
            raise ValueError(f"Model not trained for {namespace}. Run /train first.")

        with open(model_path, "rb") as f:
            models = pickle.load(f)
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError("Uploaded CSV file is empty")

        df = self.preprocess_data(df, remove_outliers_flag=False)
        if df.empty:
            raise ValueError("Preprocessed DataFrame is empty")
        
        df = df.fillna(0)
        last_hour_df = df.tail(12)

        _, job_columns = self.infer_dependency_graph(df, job_names)

        anomalies: Dict[str, Dict[str, str]] = {}
        anomaly_predictions = {}
        for job in job_names:
            job_cols = job_columns.get(job, [])
            if job_cols and job in models:
                preds = models[job].predict(last_hour_df[job_cols])
                anomaly_indices = last_hour_df.index[preds == -1]

                for idx, col in enumerate(job_cols):
                    anomaly_predictions[f"{col}_anomaly"] = preds

                if not enable_2min_duration and not enable_5min_duration:
                    if len(anomaly_indices) > 0:
                        anomalies[job] = {}
                        for col in job_cols:
                            metric_name = col.split('_', 1)[1]
                            anomaly_mask = preds == -1
                            if anomaly_mask.any():
                                first_anomaly_time = last_hour_df.index[anomaly_mask][0]
                                anomalies[job][metric_name] = str(first_anomaly_time)

        root_cause = None
        cascading_graph: Dict[str, Dict[str, Any]] = {}
        affected_jobs = []
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
                    "affected_jobs": []
                }
                visited = set([root_cause])
                queue = [root_cause]
                while queue:
                    current = queue.pop(0)
                    for succ in G.successors(current):
                        if succ in anomalies and succ not in visited:
                            cascading_graph[current]["affected_jobs"].append({
                                "job": succ,
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ]
                            })
                            cascading_graph[succ] = {
                                "first_anomaly": str(anomaly_times[succ]),
                                "metrics": anomalies[succ],
                                "affected_jobs": []
                            }
                            visited.add(succ)
                            queue.append(succ)
                            affected_jobs.append(succ)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prediction_path = self.get_prediction_path(namespace, timestamp)
        prediction_df = last_hour_df.copy()
        for col, pred_values in anomaly_predictions.items():
            prediction_df[col] = pred_values
        prediction_df.to_csv(prediction_path)
        logger.info(f"Saved prediction data to {prediction_path}")

        issue_graph_summary = None
        if root_cause:
            issue_graph_summary = self.draw_issue_graph(G, namespace, root_cause, affected_jobs)

        return {
            "status": "success",
            "anomalies": anomalies,
            "root_cause": root_cause,
            "cascading_failures_graph": cascading_graph,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_file": prediction_path,
            "issue_graph_image_url": "/issue_graph" if issue_graph_summary else None
        }

    def get_graph_image(self) -> bytes:
        """Return the dependency graph image content."""
        if not os.path.exists(self.graph_image_path):
            raise ValueError("Graph image not found. Run /train first.")
        with open(self.graph_image_path, "rb") as f:
            return f.read()

    def get_issue_graph_image(self) -> bytes:
        """Return the issue graph image content."""
        if not os.path.exists(self.issue_graph_image_path):
            raise ValueError("Issue graph image not found. Run /predict or /predict_from_csv first.")
        with open(self.issue_graph_image_path, "rb") as f:
            return f.read()

    def get_metric_graph_image(self, metric: str, namespace: str) -> bytes:
        """Return the metric behavior graph image content."""
        safe_metric_name = "".join(c if c.isalnum() or c == "_" else "_" for c in metric)
        image_path = f"{self.prediction_dir}/metric_behavior_{safe_metric_name}_{namespace}.png"
        if not os.path.exists(image_path):
            raise ValueError(f"Metric graph for {metric} not found. Run /predict first.")
        with open(image_path, "rb") as f:
            return f.read()

    def fetch_metrics(self, query: str, start_time: Optional[str], end_time: Optional[str], step: str) -> Dict[str, Any]:
        """Fetch Prometheus metrics using a custom query."""
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
