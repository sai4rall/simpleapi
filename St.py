Let’s enhance the system to persist anomaly results (including root causes and affected services) in a SQL Server database and use this persisted data for the 5-minute consistency check instead of the in-memory anomaly_history. We’ll modify the AnomalyNotifier class to store anomaly data in the database and query it for the persistence check. We’ll also update the main.py to set up the database connection.
Here’s how we’ll proceed:
	1	Set Up SQL Server Database: Define a schema to store anomaly results, root causes, and affected services.
	2	Persist Anomaly Data: Modify AnomalyNotifier to save anomaly data to the database.
	3	Update Persistence Check: Modify the 5-minute consistency check to query the database instead of using in-memory history.
	4	Integrate with Existing Code: Ensure the database operations integrate seamlessly with the existing system.
We’ll use the pyodbc library to connect to SQL Server. You’ll need to have a SQL Server instance running and a database set up with the appropriate schema.

Updated Directory Structure
project/
│
├── api.py                # FastAPI endpoints
├── anomaly_detector.py   # Business logic class
├── anomaly_notifier.py   # Anomaly persistence and email notifications (updated)
├── main.py               # Main application entry point (updated)
└── config/               # Configuration files
    ├── prometheus_queries.json
    └── service_names.json

Database Schema
We’ll create two tables in SQL Server to store the anomaly data:
	1	Anomalies: Stores the anomaly details for each service.
	◦	id: Auto-incremented primary key.
	◦	service: The service name.
	◦	metrics: JSON string of the affected metrics and their first anomaly times.
	◦	timestamp: When the anomaly was detected.
	◦	namespace: The namespace of the service.
	2	RootCausesAndAffected: Stores the root cause and affected services for each anomaly event.
	◦	id: Auto-incremented primary key.
	◦	anomaly_id: Foreign key referencing the Anomalies table.
	◦	root_cause: The root cause service.
	◦	affected_service: An affected service (one row per affected service).
Here’s the SQL to create these tables:
CREATE TABLE Anomalies (
    id INT IDENTITY(1,1) PRIMARY KEY,
    service NVARCHAR(255) NOT NULL,
    metrics NVARCHAR(MAX) NOT NULL, -- JSON string
    timestamp DATETIME NOT NULL,
    namespace NVARCHAR(255) NOT NULL
);

CREATE TABLE RootCausesAndAffected (
    id INT IDENTITY(1,1) PRIMARY KEY,
    anomaly_id INT NOT NULL,
    root_cause NVARCHAR(255),
    affected_service NVARCHAR(255),
    FOREIGN KEY (anomaly_id) REFERENCES Anomalies(id)
);

1. `anomaly_detector.py` (Unchanged)
This file remains unchanged from the previous version (as provided in your last request). It’s the same AnomalyDetector class without the anomaly persistence and email notification logic, which is now handled by AnomalyNotifier.

2. `anomaly_notifier.py` (Updated to Use SQL Server)
We’ll modify the AnomalyNotifier class to persist anomaly data in the SQL Server database and use it for the 5-minute consistency check.
# anomaly_notifier.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import smtplib
from email.mime.text import MIMEText
import pyodbc
import json

class AnomalyNotifier:
    def __init__(self, smtp_server: str, smtp_port: int, smtp_username: str, smtp_password: str, email_from: str, email_to: str, 
                 db_connection_string: str, persistence_window: int = 300):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Email configuration
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.email_from = email_from
        self.email_to = email_to

        # Database configuration
        self.db_connection_string = db_connection_string
        self.conn = pyodbc.connect(self.db_connection_string)
        self.cursor = self.conn.cursor()

        # Persistence configuration (in seconds)
        self.persistence_window = persistence_window  # Default: 300 seconds (5 minutes)

        # Track if email has been sent for a service
        self.email_sent: Dict[str, bool] = {}

    def send_email(self, service: str, metrics: Dict[str, str], first_anomaly_time: datetime) -> None:
        """Send an email notification about a persistent anomaly."""
        subject = f"Persistent Anomaly Detected in Service: {service}"
        body = f"An anomaly has persisted for {self.persistence_window // 60} minutes in service '{service}'.\n\n"
        body += f"First detected at: {first_anomaly_time}\n"
        body += "Affected Metrics:\n"
        for metric, time in metrics.items():
            body += f"- {metric}: First anomaly at {time}\n"

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.email_from
        msg['To'] = self.email_to

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            self.logger.info(f"Email sent for persistent anomaly in service {service}")
        except Exception as e:
            self.logger.error(f"Failed to send email for service {service}: {str(e)}")

    def persist_anomalies(self, anomalies: Dict[str, Dict[str, str]], root_cause: str, affected_services: List[str], namespace: str, current_time: datetime) -> None:
        """Persist anomaly data to the SQL Server database."""
        try:
            for service, metrics in anomalies.items():
                # Insert into Anomalies table
                metrics_json = json.dumps(metrics)
                self.cursor.execute(
                    """
                    INSERT INTO Anomalies (service, metrics, timestamp, namespace)
                    VALUES (?, ?, ?, ?)
                    """,
                    (service, metrics_json, current_time, namespace)
                )
                self.conn.commit()

                # Get the ID of the newly inserted anomaly
                self.cursor.execute("SELECT @@IDENTITY AS id")
                anomaly_id = self.cursor.fetchone()[0]

                # Insert into RootCausesAndAffected table
                if service == root_cause:
                    for affected_service in affected_services:
                        self.cursor.execute(
                            """
                            INSERT INTO RootCausesAndAffected (anomaly_id, root_cause, affected_service)
                            VALUES (?, ?, ?)
                            """,
                            (anomaly_id, root_cause, affected_service)
                        )
                self.conn.commit()
            self.logger.info(f"Persisted anomaly data for namespace {namespace} at {current_time}")
        except Exception as e:
            self.logger.error(f"Failed to persist anomaly data: {str(e)}")
            self.conn.rollback()

    def check_persistent_anomalies(self, service: str, metrics: Dict[str, str], namespace: str, current_time: datetime) -> None:
        """Check if an anomaly has persisted for the specified window by querying the database and send an email if so."""
        try:
            # Query anomalies for the service in the last 5 minutes
            start_time = current_time - timedelta(seconds=self.persistence_window)
            self.cursor.execute(
                """
                SELECT metrics, timestamp
                FROM Anomalies
                WHERE service = ? AND namespace = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                """,
                (service, namespace, start_time, current_time)
            )
            history = self.cursor.fetchall()

            if not history:
                return

            # Find the earliest anomaly time for this service
            earliest_time = min(pd.to_datetime(list(metrics.values())))
            for record in history:
                prev_timestamp = record.timestamp
                prev_metrics = json.loads(record.metrics)
                if (current_time - prev_timestamp).total_seconds() >= self.persistence_window:
                    # Check if the same metrics are still anomalous
                    prev_metrics_set = set(prev_metrics.keys())
                    current_metrics_set = set(metrics.keys())
                    if prev_metrics_set & current_metrics_set:  # If there is overlap in anomalous metrics
                        if service not in self.email_sent or not self.email_sent[service]:
                            self.send_email(service, metrics, earliest_time)
                            self.email_sent[service] = True
                        break
            else:
                # If the anomaly hasn't persisted for the window, reset the email sent flag
                self.email_sent[service] = False
        except Exception as e:
            self.logger.error(f"Failed to check persistent anomalies for service {service}: {str(e)}")

    def update_anomaly_history(self, anomalies: Dict[str, Dict[str, str]], root_cause: str, affected_services: List[str], namespace: str, current_time: datetime) -> None:
        """Persist anomalies to the database and check for persistence."""
        # Persist anomalies to the database
        self.persist_anomalies(anomalies, root_cause, affected_services, namespace, current_time)

        # Check for persistent anomalies
        for service, metrics in anomalies.items():
            self.check_persistent_anomalies(service, metrics, namespace, current_time)

        # Reset email sent flag for services with no anomalies
        try:
            self.cursor.execute(
                """
                SELECT DISTINCT service
                FROM Anomalies
                WHERE namespace = ? AND timestamp >= ? AND timestamp <= ?
                """,
                (namespace, current_time - timedelta(seconds=self.persistence_window), current_time)
            )
            active_services = {row.service for row in self.cursor.fetchall()}
            for service in list(self.email_sent.keys()):
                if service not in active_services:
                    self.email_sent[service] = False
        except Exception as e:
            self.logger.error(f"Failed to reset email sent flags: {str(e)}")

    def __del__(self):
        """Close the database connection when the object is destroyed."""
        if hasattr(self, 'conn'):
            self.conn.close()

3. `api.py` (Updated to Pass Additional Data to `AnomalyNotifier`)
We’ll update the api.py file to pass the root_cause, affected_services, and namespace to the AnomalyNotifier for persistence.
# api.py
from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, Optional
from anomaly_detector import AnomalyDetector
from anomaly_notifier import AnomalyNotifier
from datetime import datetime
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

def setup_routes(detector: AnomalyDetector, notifier: AnomalyNotifier):
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
            
            # Update anomaly history and check for persistence
            namespace = request.namespace if request.namespace is not None else detector.default_namespace
            notifier.update_anomaly_history(
                anomalies=result["anomalies"],
                root_cause=result["root_cause"],
                affected_services=[service["service"] for service in result["cascading_failures_graph"].get(result["root_cause"], {}).get("affected_services", [])],
                namespace=namespace,
                current_time=datetime.utcnow()
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
            
            # Update anomaly history and check for persistence
            namespace = request.namespace if request.namespace is not None else detector.default_namespace
            notifier.update_anomaly_history(
                anomalies=result["anomalies"],
                root_cause=result["root_cause"],
                affected_services=[service["service"] for service in result["cascading_failures_graph"].get(result["root_cause"], {}).get("affected_services", [])],
                namespace=namespace,
                current_time=datetime.utcnow()
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

4. `main.py` (Updated with Database Connection)
We’ll update main.py to include the database connection string and pass it to the AnomalyNotifier.
# main.py
from anomaly_detector import AnomalyDetector
from anomaly_notifier import AnomalyNotifier
from api import app, setup_routes
from apscheduler.schedulers.background import BackgroundScheduler
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

# Email configuration (example for Gmail)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@gmail.com"  # Replace with your email
SMTP_PASSWORD = "your-app-password"      # Replace with your app-specific password
EMAIL_FROM = "your-email@gmail.com"      # Replace with your email
EMAIL_TO = "recipient-email@example.com" # Replace with recipient email

# SQL Server database configuration
DB_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=your-server-name;"  # Replace with your SQL Server name
    "DATABASE=your-database-name;"  # Replace with your database name
    "UID=your-username;"  # Replace with your SQL Server username
    "PWD=your-password"  # Replace with your SQL Server password
)

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

# Initialize AnomalyNotifier
notifier = AnomalyNotifier(
    smtp_server=SMTP_SERVER,
    smtp_port=SMTP_PORT,
    smtp_username=SMTP_USERNAME,
    smtp_password=SMTP_PASSWORD,
    email_from=EMAIL_FROM,
    email_to=EMAIL_TO,
    db_connection_string=DB_CONNECTION_STRING,
    persistence_window=300  # 5 minutes
)

# Setup FastAPI routes
setup_routes(detector, notifier)

# Setup scheduler
scheduler = BackgroundScheduler()
def scheduled_predict():
    try:
        df = detector.fetch_prometheus_data(lookback="1m", services=None, namespace=None)
        if df.empty:
            detector.logger.warning("No data fetched for the last minute")
            return
        
        result = detector.predict_anomalies(
            df=df,
            services=None,
            namespace=None,
            enable_2min_duration=False,
            enable_5min_duration=False
        )
        
        # Update anomaly history and check for persistence
        notifier.update_anomaly_history(
            anomalies=result["anomalies"],
            root_cause=result["root_cause"],
            affected_services=[service["service"] for service in result["cascading_failures_graph"].get(result["root_cause"], {}).get("affected_services", [])],
            namespace=detector.default_namespace,
            current_time=datetime.utcnow()
        )
    except Exception as e:
        detector.logger.error(f"Scheduled prediction failed: {str(e)}")

# Schedule the prediction task to run every minute
scheduler.add_job(scheduled_predict, 'interval', minutes=1)
scheduler.start()

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        scheduler.shutdown()

Key Changes and Explanations
1. Database Setup
	•	Schema: Defined two tables (Anomalies and RootCausesAndAffected) to store anomaly data, root causes, and affected services.
	•	Connection: Used pyodbc to connect to SQL Server with a connection string.
2. `anomaly_notifier.py`
	•	Database Integration:
	◦	Added db_connection_string to the AnomalyNotifier constructor and initialized a pyodbc connection.
	◦	Added persist_anomalies method to save anomaly data to the database:
	▪	Inserts each anomaly into the Anomalies table.
	▪	For the root cause service, inserts related affected services into the RootCausesAndAffected table.
	◦	Updated check_persistent_anomalies to query the database for anomalies in the last 5 minutes instead of using in-memory history.
	◦	Updated update_anomaly_history to persist anomalies to the database and reset the email_sent flag by querying active services.
	•	Removed In-Memory History:
	◦	Replaced anomaly_history with database queries, ensuring persistence across application restarts.
	•	Error Handling:
	◦	Added try-except blocks to handle database errors and roll back transactions if needed.
	•	Cleanup:
	◦	Added __del__ method to close the database connection when the AnomalyNotifier object is destroyed.
3. `api.py`
	•	Updated setup_routes:
	◦	Now passes root_cause, affected_services, and namespace to notifier.update_anomaly_history.
	◦	Extracts affected_services from the cascading_failures_graph in the prediction result.
	•	Purpose: Ensures the AnomalyNotifier has all necessary data to persist anomalies and perform the persistence check.
4. `main.py`
	•	Added Database Configuration:
	◦	Added DB_CONNECTION_STRING for SQL Server.
	◦	Passed the connection string to AnomalyNotifier.
	•	Updated Scheduler:
	◦	The scheduled_predict function now passes root_cause, affected_services, and namespace to notifier.update_anomaly_history.

Dependencies
You’ll need to install the pyodbc library to connect to SQL Server:
pip install pyodbc
You’ll also need the apscheduler library (already used in the previous version):
pip install apscheduler
Additionally, ensure you have the ODBC Driver for SQL Server installed on your system. For example, on Windows, you can use the “ODBC Driver 17 for SQL Server”. On Linux, you may need to install the Microsoft ODBC Driver for SQL Server.

SQL Server Configuration
	1	Install SQL Server: Ensure you have a SQL Server instance running (e.g., SQL Server Express).
	2	Create Database: Create a database (e.g., AnomalyDB) and run the SQL script provided above to create the Anomalies and RootCausesAndAffected tables.
	3	Update Connection String:
	◦	Replace your-server-name, your-database-name, your-username, and your-password in DB_CONNECTION_STRING with your SQL Server details.
	◦	Example: DB_CONNECTION_STRING = (
	◦	    "DRIVER={ODBC Driver 17 for SQL Server};"
	◦	    "SERVER=localhost\\SQLEXPRESS;"
	◦	    "DATABASE=AnomalyDB;"
	◦	    "UID=sa;"
	◦	    "PWD=yourpassword"
	◦	)
	◦	

Example Email Configuration
For Gmail:
	•	SMTP_SERVER = "smtp.gmail.com"
	•	SMTP_PORT = 587
	•	SMTP_USERNAME = "your-email@gmail.com"
	•	SMTP_PASSWORD = "your-app-password" (generate this from Google Account settings)
	•	EMAIL_FROM = "your-email@gmail.com"
	•	EMAIL_TO = "recipient-email@example.com"
For other email providers, adjust the SMTP_SERVER and SMTP_PORT accordingly.

How It Works
	1	Persisting Anomaly Data:
	◦	After each prediction (manual or scheduled), AnomalyNotifier.persist_anomalies saves the anomaly data to the Anomalies table.
	◦	If a root cause is identified, it also saves the root cause and affected services to the RootCausesAndAffected table.
	2	Scheduled Predictions:
	◦	The scheduler in main.py runs scheduled_predict every minute.
	◦	It fetches the last 1 minute of data from Prometheus and calls predict_anomalies.
	◦	The detected anomalies are passed to notifier.update_anomaly_history.
	3	Anomaly Persistence Check:
	◦	AnomalyNotifier.check_persistent_anomalies queries the Anomalies table for anomalies in the last 5 minutes for the given service and namespace.
	◦	If the same metrics are still anomalous after 5 minutes, an email is sent (only once per anomaly episode).
	◦	The email_sent flag is reset for services that no longer have anomalies by querying the database for active services.
	4	Email Notification:
	◦	When an anomaly persists for 5 minutes, send_email sends an email with details about the service, affected metrics, and the first anomaly time.

Example Database Entries
After a prediction, the database might look like this:
Anomalies Table:
id
service
metrics
timestamp
namespace
1
app1
{“http_requests_rate”: “2025-04-07 12:00:00”}
2025-04-07 12:00:00
ring1
2
app2
{“cpu_usage”: “2025-04-07 12:00:00”}
2025-04-07 12:00:00
ring1
3
app1
{“http_requests_rate”: “2025-04-07 12:01:00”}
2025-04-07 12:01:00
ring1
RootCausesAndAffected Table:
id
anomaly_id
root_cause
affected_service
1
1
app1
app2

Example Usage
	1	Set Up the Database:
	◦	Create the database and tables using the SQL script provided.
	◦	Update the DB_CONNECTION_STRING in main.py with your SQL Server details.
	2	Train the Model: curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"namespace": "ring1", "retrain": true}'
	3	
	4	Start the Application: python main.py
	5	 The scheduler will start running and check for anomalies every minute, persisting results to the database.
	6	Receive Email Notifications:
	◦	If an anomaly persists for 5 minutes (based on database records), you’ll receive an email with details about the service and affected metrics.

Benefits of the Enhancement
	•	Persistence: Anomaly data is stored in a SQL Server database, ensuring it persists across application restarts.
	•	Scalability: Using a database allows for better scalability compared to in-memory storage, especially for large volumes of anomaly data.
	•	Querying: The database enables more complex queries for analysis (e.g., historical anomaly trends, root cause patterns).
	•	Isolation: The persistence and notification logic remains isolated in AnomalyNotifier, keeping the core AnomalyDetector unchanged.

Potential Further Improvements
	1	Database Indexing: Add indexes on the Anomalies table (e.g., on service, namespace, timestamp) to improve query performance.
	2	Historical Analysis: Add an endpoint to query historical anomaly data from the database for reporting or analysis.
	3	Batch Inserts: Optimize database inserts by batching multiple anomaly records to reduce the number of database calls.
	4	Connection Pooling: Use a connection pool (e.g., with pyodbc or SQLAlchemy) to manage database connections more efficiently.
Let me know if you’d like to explore these improvements or need further adjustments!
