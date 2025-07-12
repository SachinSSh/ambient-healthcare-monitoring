from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

app = FastAPI(title="User Health Data API", description="API for users to access their health data")

# OAuth2 setup (simplified for example)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock database functions (would connect to actual database in production)
async def get_user_data(user_id: str, data_type: Optional[str] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    """Retrieve user health data from database."""
    # This would be replaced with actual database queries
    return {"user_id": user_id, "data_type": data_type or "all", "data": "Sample health data"}

async def get_user_alerts(user_id: str, risk_level: Optional[str] = None, limit: int = 10):
    """Retrieve user health alerts from database."""
    # This would be replaced with actual database queries
    return [{"alert_id": "123", "risk_level": "MODERATE", "timestamp": datetime.now().isoformat()}]

async def update_user_preferences(user_id: str, preferences: Dict[str, Any]):
    """Update user preferences in database."""
    # This would be replaced with actual database updates
    return {"status": "success", "user_id": user_id, "preferences": preferences}

# Dependency for authorization
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # This would validate the token and return user info in production
    return {"user_id": "sample_user"}

@app.get("/health-data")
async def read_health_data(
    data_type: Optional[str] = Query(None, description="Type of health data (e.g., heart_rate, steps)"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Retrieve user's health data within a date range.
    
    Args:
        data_type: Optional type of health data to retrieve
        start_date: Optional start date for filtering data
        end_date: Optional end date for filtering data
    
    Returns:
        User health data
    """
    user_id = current_user.get("user_id")
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=7)
    
    try:
        data = await get_user_data(user_id, data_type, start_date, end_date)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving health data: {str(e)}")

@app.get("/alerts")
async def read_alerts(
    risk_level: Optional[str] = Query(None, enum=["LOW", "MODERATE", "HIGH", "CRITICAL"]),
    limit: int = Query(10, ge=1, le=100),
    current_user: Dict = Depends(get_current_user)
):
    """
    Retrieve health alerts for the current user.
    
    Args:
        risk_level: Optional filter by risk level
        limit: Maximum number of alerts to return
    
    Returns:
        List of health alerts
    """
    user_id = current_user.get("user_id")
    
    try:
        alerts = await get_user_alerts(user_id, risk_level, limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

@app.put("/preferences")
async def update_preferences(
    preferences: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Update user preferences for the monitoring system.
    
    Args:
        preferences: User preferences to update
    
    Returns:
        Confirmation of preference update
    """
    user_id = current_user.get("user_id")
    
    try:
        result = await update_user_preferences(user_id, preferences)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")

@app.post("/device-connection")
async def connect_device(
    device_info: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Connect a new health monitoring device to the user's account.
    
    Args:
        device_info: Information about the device to connect
    
    Returns:
        Confirmation of device connection
    """
    user_id = current_user.get("user_id")
    
    # Validate device info
    required_fields = ["device_id", "device_type"]
    if not all(field in device_info for field in required_fields):
        raise HTTPException(status_code=400, detail=f"Device info must contain fields: {', '.join(required_fields)}")
    
    # This would save to database in production
    device_info["user_id"] = user_id
    device_info["connected_at"] = datetime.now().isoformat()
    
    return {"status": "success", "device": device_info}
