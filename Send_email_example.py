from fastapi import FastAPI
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()

# Email configuration (replace with your own SMTP details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_FROM = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use an App Password if 2FA is enabled
EMAIL_TO = "recipient@example.com"

def send_email_after_delay():
    # Wait for 5 minutes (300 seconds)
    time.sleep(300)
    
    # Email content
    subject = "Delayed Email Notification"
    body = "This email was sent 5 minutes after the signal was received!"
    
    # Create email message
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        # Connect to SMTP server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.post("/send-signal")
async def send_signal():
    # Start the delayed email task in a separate thread
    thread = threading.Thread(target=send_email_after_delay)
    thread.start()
    
    # Return response immediately without waiting for the email
    return {"message": "Signal received! Email will be sent in 5 minutes."}

# Run with: uvicorn filename:app --reload
