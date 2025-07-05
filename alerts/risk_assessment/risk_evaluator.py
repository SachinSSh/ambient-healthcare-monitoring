from typing import Dict, Any, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
import os

class NotificationManager:
    """Manages sending notifications based on risk assessments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the notification manager."""
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'logs'
        )
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'notifications.log')
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def send_notification(self, user_id: str, risk_assessment: Dict[str, Any]) -> bool:
        """
        Send notifications based on risk assessment.
        
        Args:
            user_id: The ID of the user to notify
            risk_assessment: Risk assessment results
            
        Returns:
            bool: True if notification was sent successfully
        """
        risk_level = risk_assessment.get('overall_risk_level', 'LOW')
        
        # Skip notifications for low risk
        if risk_level == 'LOW':
            self.logger.info(f"No notification needed for user {user_id} with LOW risk level")
            return True
        
        # Get user notification preferences
        user_prefs = self._get_user_preferences(user_id)
        
        # Determine notification channels based on risk level
        channels = []
        if risk_level == 'MODERATE':
            channels = user_prefs.get('moderate_risk_channels', ['app'])
        elif risk_level == 'HIGH':
            channels = user_prefs.get('high_risk_channels', ['app', 'email'])
        elif risk_level == 'CRITICAL':
            channels = user_prefs.get('critical_risk_channels', ['app', 'email', 'sms'])
        
        # Prepare notification content
        subject = f"Health Alert: {risk_level} Risk Detected"
        message = self._format_notification_message(risk_assessment)
        
        # Send notifications through each channel
        success = True
        for channel in channels:
            if channel == 'app':
                channel_success = self._send_app_notification(user_id, subject, message, risk_level)
            elif channel == 'email':
                channel_success = self._send_email(user_prefs.get('email'), subject, message)
            elif channel == 'sms':
                channel_success = self._send_sms(user_prefs.get('phone'), message)
            else:
                self.logger.warning(f"Unknown notification channel: {channel}")
                channel_success = False
            
            success = success and channel_success
        
        return success
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user notification preferences."""
        # In a real implementation, this would fetch from a database
        # For now, return default preferences
        return {
            'moderate_risk_channels': ['app'],
            'high_risk_channels': ['app', 'email'],
            'critical_risk_channels': ['app', 'email', 'sms'],
            'email': f"{user_id}@example.com",
            'phone': "1234567890"
        }
    
    def _format_notification_message(self, risk_assessment: Dict[str, Any]) -> str:
        """Format the notification message based on risk assessment."""
        risk_level = risk_assessment.get('overall_risk_level', 'LOW')
        risk_factors = risk_assessment.get('risk_factors', [])
        recommendations = risk_assessment.get('recommendations', [])
        
        message = f"Health Alert: {risk_level} Risk Level Detected\n\n"
        
        if risk_factors:
            message += "Risk Factors:\n"
            for factor in risk_factors:
                message += f"- {factor['vital_sign'].replace('_', ' ').title()}: {factor['value']} "
                message += f"({factor['risk_level']} risk)\n"
            message += "\n"
        
        if recommendations:
            message += "Recommendations:\n"
            for recommendation in recommendations:
                message += f"- {recommendation}\n"
        
        return message
    
    def _send_app_notification(self, user_id: str, subject: str, message: str, risk_level: str) -> bool:
        """Send in-app notification."""
        try:
            # In a real implementation, this would use a push notification service
            self.logger.info(f"Sending app notification to user {user_id}: {subject}")
            
            # Simulate API call to notification service
            notification_data = {
                'user_id': user_id,
                'title': subject,
                'message': message,
                'priority': risk_level,
                'category': 'health_alert'
            }
            
            # Log instead of actually sending
            self.logger.info(f"App notification data: {notification_data}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send app notification: {e}")
            return False
    
    def _send_email(self, email_address: str, subject: str, message: str) -> bool:
        """Send email notification."""
        if not email_address:
            self.logger.warning("No email address provided")
            return False
        
        try:
            # In a real implementation, this would use an email service
            self.logger.info(f"Sending email to {email_address}: {subject}")
            
            # Log instead of actually sending
            self.logger.info(f"Email content: Subject: {subject}, Message: {message}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        if not phone_number:
            self.logger
