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
    
    def process_prometheus_data(self, query: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Process Prometheus data and create DataFrame with pod transition tracking
        
        Args:
            query: PromQL query
            start_time: Query start time
            end_time: Query end time
            
        Returns:
            Processed DataFrame
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
        
        # Find pod transitions
        pod_transitions = self.find_pod_transitions(data)
        
        # Process data into DataFrame
        records = []
        for series in data:
            metric_labels = series['metric']
            pod_name = metric_labels.get('pod', '')
            logical_pod = pod_transitions.get(pod_name, pod_name)
            
            for timestamp_str, value in series['values']:
                timestamp = float(timestamp_str)
                dt = datetime.fromtimestamp(timestamp)
                
                record = {
                    'timestamp': dt,
                    'datetime_str': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'pod_name': pod_name,
                    'logical_pod': logical_pod,
                    'value': float(value) if value != 'NaN' else None
                }
                
                # Add all other metric labels
                for key, val in metric_labels.items():
                    if key not in ['pod', '__name__']:
                        record[key] = val
                
                records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Sort by logical pod and timestamp
            df = df.sort_values(['logical_pod', 'timestamp'])
            
            # Add transition information
            df['pod_transition_detected'] = df['pod_name'] != df['logical_pod']
            
            print(f"\nPod Transitions Detected:")
            for actual_pod, logical_pod in pod_transitions.items():
                if actual_pod != logical_pod:
                    print(f"  {actual_pod} -> {logical_pod}")
        
        return df
    
    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Export DataFrame to CSV
        
        Args:
            df: DataFrame to export
            filename: Output CSV filename
        """
        if df.empty:
            print("No data to export")
            return
        
        df.to_csv(filename, index=False)
        print(f"\nExported {len(df)} records to {filename}")
        
        # Print summary
        unique_pods = df['pod_name'].nunique()
        unique_logical_pods = df['logical_pod'].nunique()
        print(f"Summary:")
        print(f"  - Unique actual pods: {unique_pods}")
        print(f"  - Unique logical pods: {unique_logical_pods}")
        print(f"  - Pod transitions detected: {unique_pods - unique_logical_pods}")

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
            'filename': f'cpu_usage_{NAMESPACE}_{int(time.time())}.csv'
        },
        {
            'name': 'memory_usage', 
            'query': f'sum by (pod,job) (process_resident_memory_bytes{{namespace="{NAMESPACE}"}})',
            'filename': f'memory_usage_{NAMESPACE}_{int(time.time())}.csv'
        }
    ]
    
    # Process each query
    for query_config in queries:
        print(f"\n{'='*50}")
        print(f"Processing: {query_config['name']}")
        print(f"{'='*50}")
        
        df = exporter.process_prometheus_data(
            query_config['query'], 
            start_time, 
            end_time
        )
        
        if not df.empty:
            exporter.export_to_csv(df, query_config['filename'])
            
            # Display sample data
            print(f"\nSample data preview:")
            print(df.head(10).to_string())
        else:
            print(f"No data found for {query_config['name']}")

if __name__ == "__main__":
    main()
