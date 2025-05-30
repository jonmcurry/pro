### src/monitoring/notifications.py
"""
Email Notification System
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional
import threading
from datetime import datetime


class EmailNotifier:
    """Email notification system for alerts and status updates."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # SMTP configuration
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        
        # Notification settings
        self.enabled = config.get('enabled', False)
        self.recipients = config.get('recipients', [])
        self.severity_levels = config.get('severity_levels', ['ERROR', 'CRITICAL'])
        
        self._send_lock = threading.Lock()
    
    def send_error_notification(self, subject: str, message: str, severity: str = 'ERROR'):
        """Send error notification email."""
        if not self._should_send_notification(severity):
            return
            
        try:
            with self._send_lock:
                self._send_email(
                    subject=f"[EDI Processing {severity}] {subject}",
                    body=self._format_error_message(message, severity),
                    recipients=self.recipients
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send error notification: {str(e)}")
    
    def send_status_notification(self, subject: str, status_data: Dict[str, Any]):
        """Send processing status notification."""
        if not self.enabled:
            return
            
        try:
            with self._send_lock:
                self._send_email(
                    subject=f"[EDI Processing Status] {subject}",
                    body=self._format_status_message(status_data),
                    recipients=self.recipients
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send status notification: {str(e)}")
    
    def send_completion_notification(self, processing_stats: Dict[str, Any]):
        """Send processing completion notification."""
        if not self.enabled:
            return
            
        try:
            subject = "EDI Claims Processing Completed"
            body = self._format_completion_message(processing_stats)
            
            with self._send_lock:
                self._send_email(subject, body, self.recipients)
                
        except Exception as e:
            self.logger.error(f"Failed to send completion notification: {str(e)}")
    
    def _should_send_notification(self, severity: str) -> bool:
        """Check if notification should be sent based on severity."""
        return self.enabled and severity in self.severity_levels
    
    def _send_email(self, subject: str, body: str, recipients: List[str]):
        """Send email using SMTP."""
        if not recipients or not self.smtp_server:
            return
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully: {subject}")
            
        except Exception as e:
            self.logger.error(f"SMTP error: {str(e)}")
            raise
    
    def _format_error_message(self, message: str, severity: str) -> str:
        """Format error message for email."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_template = f"""
        <html>
        <body>
            <h2 style="color: red;">EDI Processing Error</h2>
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Error Details:</strong></p>
            <pre style="background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd;">
{message}
            </pre>
            <p>Please check the system logs for more details.</p>
        </body>
        </html>
        """
        
        return html_template
    
    def _format_status_message(self, status_data: Dict[str, Any]) -> str:
        """Format status message for email."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        status_items = []
        for key, value in status_data.items():
            status_items.append(f"<li><strong>{key}:</strong> {value}</li>")
        
        html_template = f"""
        <html>
        <body>
            <h2 style="color: blue;">EDI Processing Status Update</h2>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <h3>Status Information:</h3>
            <ul>
                {''.join(status_items)}
            </ul>
        </body>
        </html>
        """
        
        return html_template
    
    def _format_completion_message(self, stats: Dict[str, Any]) -> str:
        """Format completion message for email."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_template = f"""
        <html>
        <body>
            <h2 style="color: green;">EDI Claims Processing Completed Successfully</h2>
            <p><strong>Completion Time:</strong> {timestamp}</p>
            
            <h3>Processing Statistics:</h3>
            <table border="1" style="border-collapse: collapse;">
                <tr><td><strong>Total Claims Processed</strong></td><td>{stats.get('total_claims', 'N/A')}</td></tr>
                <tr><td><strong>Processing Duration</strong></td><td>{stats.get('duration', 'N/A')} seconds</td></tr>
                <tr><td><strong>Average Rate</strong></td><td>{stats.get('rate', 'N/A')} claims/hour</td></tr>
                <tr><td><strong>Successful Validations</strong></td><td>{stats.get('successful', 'N/A')}</td></tr>
                <tr><td><strong>Failed Validations</strong></td><td>{stats.get('failed', 'N/A')}</td></tr>
                <tr><td><strong>Error Count</strong></td><td>{stats.get('errors', 'N/A')}</td></tr>
            </table>
            
            <p>Detailed logs are available on the processing server.</p>
        </body>
        </html>
        """
        
        return html_template