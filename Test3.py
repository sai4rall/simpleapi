import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class PrometheusCSVExporter:
    def __init__(self, prometheus_url: str, step: str = '30s'):
        """
        Initialize Prometheus CSV Exporter
        
        Args:
            prometheus_url: Base URL of Prometheus server (e.g., 'http://localhost:9090')
            step: Query step interval (e.g., '30s', '1m')
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.step = step
        self.pod_transitions = {}  # Track pod lifecycle transitions
        self.all_data = pd.DataFrame()  # Store all combined data
        
    def query_range(self, query: str, start_time: datetime, end_time: datetime) -> dict:
        """
        Execute Prometheus range query
        
        Args:
            query: PromQL query string
            start_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            JSON response from Prometheus API
        """
        url = f"{self.prometheus_url}/api/v1/query_range"
        
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': self.step
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Prometheus: {e}")
            return {}
    
    def extract_pod_name_pattern(self, pod_name: str) -> str:
        """
        Extract base pod name pattern (removing replica set hash and pod hash)
        
        Args:
            pod_name: Full pod name
            
        Returns:
            Base pod name pattern
        """
        # Remove common Kubernetes suffixes like replica set hash and pod hash
        # Example: myapp-deployment-7d4b8c9f-xyz123 -> myapp-deployment
        pattern = re.sub(r'-[a-f0-9]{8,10}-[a-z0-9]{5}$', '', pod_name)  # Deployment
        pattern = re.sub(r'-[a-z0-9]{5}$', '', pattern)  # ReplicaSet
        return pattern
    
    def find_pod_transitions(self, data: List[dict], timeline_gap_threshold: int = 120) -> Dict[str, str]:
        """
        Identify pod transitions based on metric gaps and timeline analysis
        
        Args:
            data: Prometheus metric data for pods
            timeline_gap_threshold: Gap in seconds to consider pod died (default: 2 minutes)
            
        Returns:
            Dictionary mapping new pod names to logical pod identifiers
        """
        pod_timelines = defaultdict(list)
        pod_transitions = {}
        
        # Build timeline for each pod
        for series in data:
            pod_name = series['metric'].get('pod', '')
            if not pod_name:
                continue
                
            timestamps = []
            for value in series['values']:
                timestamp = float(value[0])
                timestamps.append(timestamp)
            
            if timestamps:
                pod_timelines[pod_name] = {
                    'start': min(timestamps),
                    'end': max(timestamps),
                    'timestamps': sorted(timestamps),
                    'base_pattern': self.extract_pod_name_pattern(pod_name)
                }
        
        # Group pods by base pattern
        pattern_groups = defaultdict(list)
        for pod_name, timeline in pod_timelines.items():
            pattern_groups[timeline['base_pattern']].append((pod_name, timeline))
        
        # Identify transitions within each pattern group
        logical_pod_counter = {}
        for base_pattern, pods in pattern_groups.items():
            # Sort pods by start time
            pods.sort(key=lambda x: x[1]['start'])
            
            logical_pod_id = f"{base_pattern}-logical-1"
            logical_pod_counter[base_pattern] = 1
            
            for i, (pod_name, timeline) in enumerate(pods):
                if i == 0:
                    # First pod in the group
                    pod_transitions[pod_name] = logical_pod_id
                else:
                    # Check if this pod started shortly after previous pod(s) ended
                    pod_start = timeline['start']
                    transition_found = False
                    
                    for j in range(i):
                        prev_pod_name, prev_timeline = pods[j]
                        prev_end = prev_timeline['end']
                        
                        # Check if current pod started within threshold after previous pod ended
                        time_gap = pod_start - prev_end
                        if 0 <= time_gap <= timeline_gap_threshold:
                            # This is likely a replacement pod
                            prev_logical_id = pod_transitions[prev_pod_name]
                            pod_transitions[pod_name] = prev_logical_id
                            transition_found = True
                            break
                    
                    if not transition_found:
                        # This is a new logical pod
                        logical_pod_counter[base_pattern] += 1
                        logical_pod_id = f"{base_pattern}-logical-{logical_pod_counter[base_pattern]}"
                        pod_transitions[pod_name] = logical_pod_id
        
        return pod_transitions
    
    def generate_column_name(self, metric_labels: dict, metric_name: str, group_by_labels: list) -> str:
        """
        Generate dynamic column name based on grouped labels and metric name
        
        Args:
            metric_labels: Dictionary of metric labels
            metric_name: Name of the metric (e.g., 'cpu_usage', 'memory_usage')
            group_by_labels: List of labels used in group by clause
            
        Returns:
            Generated column name
        """
        # Extract values for group by labels
        label_values = []
        for label in group_by_labels:
            value = metric_labels.get(label, 'unknown')
            # Clean the value to make it filesystem/column safe
            clean_value = re.sub(r'[^a-zA-Z0-9_-]', '_', str(value))
            label_values.append(f"{label}{clean_value}")
        
        # Combine all parts
        if label_values:
            column_name = "_".join(label_values) + f"_{metric_name}"
        else:
            column_name = metric_name
            
        return column_name
    
    def extract_group_by_labels(self, query: str) -> list:
        """
        Extract group by labels from PromQL query
        
        Args:
            query: PromQL query string
            
        Returns:
            List of group by labels
        """
        # Look for patterns like "sum by (pod,job)" or "rate by (instance, job)"
        pattern = r'by\s*\(\s*([^)]+)\s*\)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            labels_str = match.group(1)
            # Split by comma and clean up whitespace
            labels = [label.strip() for label in labels_str.split(',')]
            return labels
        
        return []
    
    def process_prometheus_data(self, query: str, metric_name: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Process Prometheus data and create DataFrame with dynamic column naming
        
        Args:
            query: PromQL query
            metric_name: Name for this metric (e.g., 'cpu_usage')
            start_time: Query start time
            end_time: Query end time
            
        Returns:
            Processed DataFrame with dynamic columns
        """
        print(f"Executing query: {query}")
        response = self.query_range(query, start_time, end_time)
        
        if not response or response.get('status') != 'success':
            print("Failed to get valid response from Prometheus")
            return pd.DataFrame()
        
        data = response['data']['result']
        if not data:
            print("No data returned from query")
            return pd.DataFrame()
        
        # Extract group by labels from query
        group_by_labels = self.extract_group_by_labels(query)
        print(f"Detected group by labels: {group_by_labels}")
        
        # Find pod transitions
        pod_transitions = self.find_pod_transitions(data)
        
        # Process data into DataFrame with dynamic column naming
        records_by_timestamp = defaultdict(dict)
        
        for series in data:
            metric_labels = series['metric']
            pod_name = metric_labels.get('pod', '')
            logical_pod = pod_transitions.get(pod_name, pod_name)
            
            # Generate dynamic column name
            column_name = self.generate_column_name(metric_labels, metric_name, group_by_labels)
            
            for timestamp_str, value in series['values']:
                timestamp = float(timestamp_str)
                dt = datetime.fromtimestamp(timestamp)
                
                # Use logical pod in the record key to handle transitions
                record_key = (dt, logical_pod if pod_name else dt)
                
                if record_key not in records_by_timestamp:
                    records_by_timestamp[record_key] = {
                        'timestamp': dt,
                        'datetime_str': dt.strftime('%Y-%m-%d %H:%M:%S'),
                        'logical_pod': logical_pod if pod_name else None
                    }
                
                # Add the metric value with dynamic column name
                records_by_timestamp[record_key][column_name] = float(value) if value != 'NaN' else None
        
        # Convert to DataFrame
        records = list(records_by_timestamp.values())
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            print(f"Generated {len(df)} records with columns: {[col for col in df.columns if col not in ['timestamp', 'datetime_str', 'logical_pod']]}")
            
            print(f"\nPod Transitions Detected for {metric_name}:")
            transition_count = 0
            for actual_pod, logical_pod in pod_transitions.items():
                if actual_pod != logical_pod:
                    print(f"  {actual_pod} -> {logical_pod}")
                    transition_count += 1
            
            if transition_count == 0:
                print("  No pod transitions detected")
        
        return df
    
    def merge_dataframes(self, new_df: pd.DataFrame):
        """
        Merge new DataFrame with existing combined data
        
        Args:
            new_df: New DataFrame to merge
        """
        if new_df.empty:
            return
        
        if self.all_data.empty:
            self.all_data = new_df.copy()
        else:
            # Merge on timestamp and logical_pod
            merge_columns = ['timestamp', 'datetime_str']
            if 'logical_pod' in new_df.columns and 'logical_pod' in self.all_data.columns:
                merge_columns.append('logical_pod')
            
            self.all_data = pd.merge(
                self.all_data, 
                new_df, 
                on=merge_columns, 
                how='outer'
            )
        
        # Sort by timestamp and logical_pod
        sort_columns = ['timestamp']
        if 'logical_pod' in self.all_data.columns:
            sort_columns.append('logical_pod')
        
        self.all_data = self.all_data.sort_values(sort_columns)
    
    def export_to_csv(self, filename: str):
        """
        Export combined DataFrame to CSV
        
        Args:
            filename: Output CSV filename
        """
        if self.all_data.empty:
            print("No data to export")
            return
        
        # Reorder columns - put timestamp and logical_pod first
        columns = ['timestamp', 'datetime_str']
        if 'logical_pod' in self.all_data.columns:
            columns.append('logical_pod')
        
        # Add all metric columns
        metric_columns = [col for col in self.all_data.columns if col not in columns]
        columns.extend(sorted(metric_columns))
        
        # Reorder DataFrame
        self.all_data = self.all_data[columns]
        
        self.all_data.to_csv(filename, index=False)
        print(f"\nExported {len(self.all_data)} records to {filename}")
        
        # Print summary
        metric_columns = [col for col in self.all_data.columns if col not in ['timestamp', 'datetime_str', 'logical_pod']]
        unique_logical_pods = self.all_data['logical_pod'].nunique() if 'logical_pod' in self.all_data.columns else 0
        
        print(f"Summary:")
        print(f"  - Total records: {len(self.all_data)}")
        print(f"  - Unique logical pods: {unique_logical_pods}")
        print(f"  - Metric columns: {len(metric_columns)}")
        print(f"  - Metric column names: {metric_columns}")

def main():
    # Configuration
    PROMETHEUS_URL = "http://localhost:9090"  # Update with your Prometheus URL
    NAMESPACE = "test"  # Update with your namespace
    
    # Initialize exporter
    exporter = PrometheusCSVExporter(PROMETHEUS_URL, step='30s')
    
    # Define time range (last 4 hours as example)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=4)
    
    # Example queries - modify as needed
    queries = [
        {
            'name': 'cpu_usage',
            'query': f'sum by (pod,job) (rate(process_cpu_seconds_total{{namespace="{NAMESPACE}"}}[5m]))',
        },
        {
            'name': 'memory_usage', 
            'query': f'sum by (pod,job) (process_resident_memory_bytes{{namespace="{NAMESPACE}"}})',
        },
        {
            'name': 'network_rx',
            'query': f'sum by (pod,job) (rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}"}}[5m]))',
        },
        {
            'name': 'disk_usage',
            'query': f'sum by (pod,job,device) (container_fs_usage_bytes{{namespace="{NAMESPACE}"}})',
        }
    ]
    
    # Process each query and merge into single dataset
    print(f"{'='*70}")
    print(f"Processing {len(queries)} queries for namespace: {NAMESPACE}")
    print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    for i, query_config in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query_config['name']}")
        print(f"Query: {query_config['query']}")
        
        df = exporter.process_prometheus_data(
            query_config['query'], 
            query_config['name'],
            start_time, 
            end_time
        )
        
        # Merge with combined dataset
        exporter.merge_dataframes(df)
        
        print(f"✓ Processed {query_config['name']}")
    
    # Export combined data to single CSV
    output_filename = f'prometheus_metrics_{NAMESPACE}_{int(time.time())}.csv'
    exporter.export_to_csv(output_filename)
    
    # Display sample of combined data
    if not exporter.all_data.empty:
        print(f"\nSample of combined data (first 5 rows):")
        print("="*100)
        sample_df = exporter.all_data.head()
        
        # Display in a more readable format
        for idx, row in sample_df.iterrows():
            print(f"\nRow {idx + 1}:")
            print(f"  Timestamp: {row['datetime_str']}")
            if 'logical_pod' in row:
                print(f"  Logical Pod: {row['logical_pod']}")
            
            # Show metric values
            metric_cols = [col for col in row.index if col not in ['timestamp', 'datetime_str', 'logical_pod']]
            for col in metric_cols:
                if pd.notna(row[col]):
                    print(f"  {col}: {row[col]}")

if __name__ == "__main__":
    main()
