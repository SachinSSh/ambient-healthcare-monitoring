import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class HealthInsightGenerator:
    """
    Analyzes health data to generate insights and recommendations.
    Uses statistical analysis and machine learning to identify patterns and anomalies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the health insight generator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'analysis_config.json'
        )
        
        # Load configuration
        self.config = self._load_or_create_config()
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Initialize models
        self.anomaly_detectors = {}
        self._initialize_models()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the health insight generator."""
        logger = logging.getLogger('health_insight_generator')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(logs_dir, 'health_insights.log'))
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create a new one if it doesn't exist."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        else:
            # Create a new config
            config = {
                'data_types': {
                    'heart_rate': {
                        'normal_range': [60, 100],
                        'unit': 'bpm',
                        'anomaly_detection': {
                            'enabled': True,
                            'method': 'isolation_forest',
                            'contamination': 0.05
                        },
                        'trend_analysis': {
                            'enabled': True,
                            'window_size': 24  # hours
                        }
                    },
                    'temperature': {
                        'normal_range': [36.1, 37.2],
                        'unit': '°C',
                        'anomaly_detection': {
                            'enabled': True,
                            'method': 'isolation_forest',
                            'contamination': 0.05
                        },
                        'trend_analysis': {
                            'enabled': True,
                            'window_size': 24  # hours
                        }
                    },
                    'blood_oxygen': {
                        'normal_range': [95, 100],
                        'unit': '%',
                        'anomaly_detection': {
                            'enabled': True,
                            'method': 'isolation_forest',
                            'contamination': 0.05
                        },
                        'trend_analysis': {
                            'enabled': True,
                            'window_size': 24  # hours
                        }
                    },
                    'glucose': {
                        'normal_range': [70, 140],
                        'unit': 'mg/dL',
                        'anomaly_detection': {
                            'enabled': True,
                            'method': 'isolation_forest',
                            'contamination': 0.05
                        },
                        'trend_analysis': {
                            'enabled': True,
                            'window_size': 24  # hours
                        }
                    }
                },
                'correlation_analysis': {
                    'enabled': True,
                    'min_correlation': 0.7
                },
                'pattern_detection': {
                    'enabled': True,
                    'methods': ['daily_patterns', 'weekly_patterns']
                },
                'insight_generation': {
                    'max_insights': 5,
                    'prioritize_anomalies': True
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save the config
            with open(self.config_path, 'w') as config_file:
                json.dump(config, config_file, indent=4)
            
            return config
    
    def _initialize_models(self) -> None:
        """Initialize machine learning models for anomaly detection."""
        for data_type, config in self.config['data_types'].items():
            if config.get('anomaly_detection', {}).get('enabled', False):
                method = config['anomaly_detection'].get('method', 'isolation_forest')
                
                if method == 'isolation_forest':
                    contamination = config['anomaly_detection'].get('contamination', 0.05)
                    self.anomaly_detectors[data_type] = IsolationForest(
                        contamination=contamination,
                        random_state=42
                    )
                    self.logger.info(f"Initialized Isolation Forest for {data_type}")
    
    def generate_insights(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate health insights from the provided data.
        
        Args:
            data: Dictionary mapping data types to lists of readings
                Each reading should have 'timestamp' and 'value' keys
                
        Returns:
            Dictionary containing insights, anomalies, trends, and recommendations
        """
        self.logger.info("Generating health insights")
        
        # Convert data to DataFrames
        dataframes = {}
        for data_type, readings in data.items():
            if not readings:
                continue
                
            df = pd.DataFrame(readings)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            dataframes[data_type] = df
        
        # Generate insights
        results = {
            'insights': [],
            'anomalies': [],
            'trends': [],
            'correlations': [],
            'recommendations': []
        }
        
        # Process each data type
        for data_type, df in dataframes.items():
            if data_type not in self.config['data_types']:
                continue
                
            data_config = self.config['data_types'][data_type]
            
            # Check for anomalies
            if data_config.get('anomaly_detection', {}).get('enabled', False):
                anomalies = self._detect_anomalies(data_type, df)
                results['anomalies'].extend(anomalies)
                
                # Generate insights from anomalies
                for anomaly in anomalies:
                    insight = {
                        'type': 'anomaly',
                        'data_type': data_type,
                        'description': anomaly['description'],
                        'severity': anomaly['severity'],
                        'timestamp': anomaly['timestamp']
                    }
                    results['insights'].append(insight)
            
            # Analyze trends
            if data_config.get('trend_analysis', {}).get('enabled', False):
                trends = self._analyze_trends(data_type, df)
                results['trends'].extend(trends)
                
                # Generate insights from trends
                for trend in trends:
                    if trend['significant']:
                        insight = {
                            'type': 'trend',
                            'data_type': data_type,
                            'description': trend['description'],
                            'severity': 'info' if trend['direction'] == 'stable' else 'warning',
                            'timestamp': trend['end_time']
                        }
                        results['insights'].append(insight)
        
        # Analyze correlations between data types
        if self.config.get('correlation_analysis', {}).get('enabled', False) and len(dataframes) > 1:
            correlations = self._analyze_correlations(dataframes)
            results['correlations'] = correlations
            
            # Generate insights from correlations
            for correlation in correlations:
                if correlation['significant']:
                    insight = {
                        'type': 'correlation',
                        'data_types': [correlation['data_type_1'], correlation['data_type_2']],
                        'description': correlation['description'],
                        'severity': 'info',
                        'timestamp': datetime.now().isoformat()
                    }
                    results['insights'].append(insight)
        
        # Detect patterns
        if self.config.get('pattern_detection', {}).get('enabled', False):
            patterns = self._detect_patterns(dataframes)
            
            # Generate insights from patterns
            for pattern in patterns:
                insight = {
                    'type': 'pattern',
                    'data_type': pattern['data_type'],
                    'description': pattern['description'],
                    'severity': 'info',
                    'timestamp': datetime.now().isoformat()
                }
                results['insights'].append(insight)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Limit number of insights
        max_insights = self.config.get('insight_generation', {}).get('max_insights', 5)
        prioritize_anomalies = self.config.get('insight_generation', {}).get('prioritize_anomalies', True)
        
        if prioritize_anomalies:
            # Sort by severity and type (anomalies first)
            severity_order = {'critical': 0, 'warning': 1, 'info': 2}
            type_order = {'anomaly': 0, 'trend': 1, 'correlation': 2, 'pattern': 3}
            
            results['insights'].sort(key=lambda x: (
                severity_order.get(x['severity'], 3),
                type_order.get(x['type'], 4)
            ))
        else:
            # Sort by severity only
            severity_order = {'critical': 0, 'warning': 1, 'info': 2}
            results['insights'].sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        results['insights'] = results['insights'][:max_insights]
        
        self.logger.info(f"Generated {len(results['insights'])} insights, {len(results['recommendations'])} recommendations")
        
        return results
    
    def _detect_anomalies(self, data_type: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data using machine learning.
        
        Args:
            data_type: Type of health data
            df: DataFrame containing the data
            
        Returns:
            List of detected anomalies
        """
        if len(df) < 10:  # Need enough data for anomaly detection
            return []
            
        anomalies = []
        data_config = self.config['data_types'][data_type]
        normal_range = data_config.get('normal_range', [0, 100])
        unit = data_config.get('unit', '')
        
        # Simple range-based anomaly detection
        for _, row in df.iterrows():
            value = row['value']
            if value < normal_range[0] or value > normal_range[1]:
                severity = 'warning'
                if value < normal_range[0] * 0.9 or value > normal_range[1] * 1.1:
                    severity = 'critical'
                    
                anomalies.append({
                    'data_type': data_type,
                    'timestamp': row['timestamp'].isoformat(),
                    'value': value,
                    'expected_range': normal_range,
                    'description': f"Abnormal {data_type.replace('_', ' ')} reading: {value}{unit} (normal range: {normal_range[0]}-{normal_range[1]}{unit})",
                    'severity': severity
                })
        
        # Machine learning-based anomaly detection
        if data_type in self.anomaly_detectors and len(df) >= 10:
            # Prepare data
            X = df['value'].values.reshape(-1, 1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit and predict
            model = self.anomaly_detectors[data_type]
            model.fit(X_scaled)
            predictions = model.predict(X_scaled)
            
            # Find anomalies (predictions of -1)
            for i, pred in enumerate(predictions):
                if pred == -1 and df.iloc[i]['value'] not in [a['value'] for a in anomalies]:
                    value = df.iloc[i]['value']
                    timestamp = df.iloc[i]['timestamp']
                    
                    # Determine severity based on distance from normal range
                    severity = 'warning'
                    if (value < normal_range[0] * 0.8 or value > normal_range[1] * 1.2):
                        severity = 'critical'
                    
                    anomalies.append({
                        'data_type': data_type,
                        'timestamp': timestamp.isoformat(),
                        'value': value,
                        'expected_range': normal_range,
                        'description': f"Unusual {data_type.replace('_', ' ')} pattern detected: {value}{unit}",
                        'severity': severity,
                        'ml_detected': True
                    })
        
        return anomalies
    
    def _analyze_trends(self, data_type: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze trends in the data.
        
        Args:
            data_type: Type of health data
            df: DataFrame containing the data
            
        Returns:
            List of detected trends
        """
        if len(df) < 5:  # Need enough data for trend analysis
            return []
            
        trends = []
        data_config = self.config['data_types'][data_type]
        window_size = data_config.get('trend_analysis', {}).get('window_size', 24)
        unit = data_config.get('unit', '')
        
        # Convert window size from hours to timedelta
        window_delta = timedelta(hours=window_size)
        
        # Get the latest timestamp
        latest_time = df['timestamp'].max()
        
        # Filter data within the window
        window_start = latest_time - window_delta
        window_data = df[df['timestamp'] >= window_start]
        
        if len(window_data) < 3:  # Need at least 3 points for trend
            return []
        
        # Calculate trend using linear regression
        x = np.array((window_data['timestamp'] - window_start).dt.total_seconds()).reshape(-1, 1)
        y = window_data['value'].values
        
        if len(x) == 0 or len(y) == 0:
            return []
            
        # Simple linear regression
        slope, intercept = np.polyfit(x.flatten(), y, 1)
        
        # Determine trend direction and significance
        direction = 'stable'
        significant = False
        
        # Calculate average value
        avg_value = np.mean(y)
        
        # Calculate percent change over the window
        total_change_percent = (slope * x[-1][0]) / avg_value * 100
        
        if abs(total_change_percent) >= 5:  # 5% change is significant
            significant = True
            if slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        
        # Create trend description
        if significant:
            description = f"{data_type.replace('_', ' ')} is {direction} by approximately {abs(total_change_percent):.1f}% over the past {window_size} hours"
        else:
            description = f"{data_type.replace('_', ' ')} has been stable over the past {window_size} hours"
        
        trends.append({
            'data_type': data_type,
            'start_time': window_start.isoformat(),
            'end_time': latest_time.isoformat(),
            'direction': direction,
            'slope': slope,
            'percent_change': total_change_percent,
            'significant': significant,
            'description': description
        })
        
        return trends
    
    def _analyze_correlations(self, dataframes: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Analyze correlations between different data types.
        
        Args:
            dataframes: Dictionary mapping data types to DataFrames
            
        Returns:
            List of detected correlations
        """
        correlations = []
        min_correlation = self.config.get('correlation_analysis', {}).get('min_correlation', 0.7)
        
        # Get all pairs of data types
        data_types = list(dataframes.keys())
        for i in range(len(data_types)):
            for j in range(i + 1, len(data_types)):
                data_type_1 = data_types[i]
                data_type_2 = data_types[j]
                
                df1 = dataframes[data_type_1]
                df2 = dataframes[data_type_2]
                
                # Need enough data points for correlation
                if len(df1) < 5 or len(df2) < 5:
                    continue
                
                # Merge dataframes on timestamp
                df1_resampled = df1.set_index('timestamp').resample('1H').mean().reset_index()
                df2_resampled = df2.set_index('timestamp').resample('1H').mean().reset_index()
                
                merged = pd.merge_asof(
                    df1_resampled.sort_values('timestamp'),
                    df2_resampled.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    suffixes=('_1', '_2')
                )
                
                # Need enough overlapping data points
                if len(merged) < 5:
                    continue
                
                # Calculate correlation
                correlation = merged['value_1'].corr(merged['value_2'])
                
                # Check if correlation is significant
                significant = abs(correlation) >= min_correlation
                
                if significant:
                    direction = 'positively' if correlation > 0 else 'negatively'
                    description = f"{data_type_1.replace('_', ' ')} and {data_type_2.replace('_', ' ')} are {direction} correlated (r={correlation:.2f})"
                    
                    correlations.append({
                        'data_type_1': data_type_1,
                        'data_type_2': data_type_2,
                        'correlation': correlation,
                        'significant': significant,
                        'description': description
                    })
        
        return correlations
    
    def _detect_patterns(self, dataframes: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Detect patterns in the data.
        
        Args:
            dataframes: Dictionary mapping data types to DataFrames
            
        Returns:
            List of detected patterns
        """
        patterns = []
        methods = self.config.get('pattern_detection', {}).get('methods', [])
        
        for data_type, df in dataframes.items():
            if len(df) < 24:  # Need at least a day of data
                continue
                
            # Add hour and day of week columns
            df_with_time = df.copy()
            df_with_time['hour'] = df_with_time['timestamp'].dt.hour
            df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek
            
            # Detect daily patterns
            if 'daily_patterns' in methods:
                daily_pattern = self._detect_daily_pattern(data_type, df_with_time)
                if daily_pattern:
                    patterns.append(daily_pattern)
            
            # Detect weekly patterns
            if 'weekly_patterns' in methods and len(df) >= 168:  # Need at least a week of data
                weekly_pattern = self._detect_weekly_pattern(data_type, df_with_time)
                if weekly_pattern:
                    patterns.append(weekly_pattern)
        
        return patterns
    
    def _detect_daily_pattern(self, data_type: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect daily patterns in the data.
        
        Args:
            data_type: Type of health data
            df: DataFrame containing the data with hour column
            
        Returns:
            Detected pattern or None
        """
        # Group by hour and calculate mean
        hourly_means = df.groupby('hour')['value'].mean()
        
        if len(hourly_means) < 12:  # Need at least half a day of data
            return None
        
        # Find peak and trough hours
        peak_hour = hourly_means.idxmax()
        trough_hour = hourly_means.idxmin()
        
        # Calculate variation
        variation = hourly_means.max() - hourly_means.min()
        mean_value = hourly_means.mean()
        variation_percent = (variation / mean_value) * 100
        
        # Check if variation is significant
        if variation_percent < 10:  # Less than 10% variation is not significant
            return None
        
        # Format hours for readability
        peak_hour_str = f"{peak_hour:02d}:00"
        trough_hour_str = f"{trough_hour:02d}:00"
        
        description = f"{data_type.replace('_', ' ')} shows a daily pattern with peak at {peak_hour_str} and lowest at {trough_hour_str} (variation: {variation_percent:.1f}%)"
        
        return {
            'data_type': data_type,
            'pattern_type': 'daily',
            'peak_hour': peak_hour,
            'trough_hour': trough_hour,
            'variation_percent': variation_percent,
            'description': description
        }
    
    def _detect_weekly_pattern(self, data_type: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect weekly patterns in the data.
        
        Args:
            data_type: Type of health data
            df: DataFrame containing the data with day_of_week column
            
        Returns:
            Detected pattern or None
        """
        # Group by day of week and calculate mean
        daily_means = df.groupby('day_of_week')['value'].mean()
        
        if len(daily_means) < 5:  # Need at least 5 days of data
            return None
        
        # Find peak and trough days
        peak_day = daily_means.idxmax()
        trough_day = daily_means.idxmin()
        
        # Calculate variation
        variation = daily_means.max() - daily_means.min()
        mean_value = daily_means.mean()
        variation_percent = (variation / mean_value) * 100
        
        # Check if variation is significant
        if variation_percent < 10:  # Less than 10% variation is not significant
            return None
        
        # Convert day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_day_name = day_names[peak_day]
        trough_day_name = day_names[trough_day]
        
        description = f"{data_type.replace('_', ' ')} shows a weekly pattern with highest on {peak_day_name} and lowest on {trough_day_name} (variation: {variation_percent:.1f}%)"
        
        return {
            'data_type': data_type,
            'pattern_type': 'weekly',
            'peak_day': peak_day,
            'trough_day': trough_day,
            'variation_percent': variation_percent,
            'description': description
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on insights.
        
        Args:
            results: Dictionary containing insights, anomalies, trends, etc.
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for critical anomalies
        critical_anomalies = [a for a in results['anomalies'] if a['severity'] == 'critical']
        if critical_anomalies:
            recommendations.append("Contact your healthcare provider immediately about the critical readings detected.")
        
        # Check for warning anomalies
        warning_anomalies = [a for a in results['anomalies'] if a['severity'] == 'warning']
        if warning_anomalies:
            recommendations.append("Monitor your health closely and consider consulting your healthcare provider about the abnormal readings.")
        
        # Check for concerning trends
        concerning_trends = [t for t in results['trends'] if t['significant'] and t['direction'] != 'stable']
        if concerning_trends:
            for trend in concerning_trends:
                data_type = trend['data_type']
                direction = trend['direction']
                
                if data_type == 'heart_rate':
                    if direction == 'increasing':
                        recommendations.append("Your heart rate has been increasing. Consider reducing caffeine intake and practicing relaxation techniques.")
                    else:
                        recommendations.append("Your heart rate has been decreasing. Ensure you're staying hydrated and getting adequate rest.")
                
                elif data_type == 'temperature':
                    if direction == 'increasing':
                        recommendations.append("Your body temperature has been rising. Stay hydrated and monitor for other symptoms of infection.")
                    else:
                        recommendations.append("Your body temperature has been decreasing. Ensure you're staying warm and monitor for any unusual symptoms.")
                
                elif data_type == 'blood_oxygen':
                    if direction == 'decreasing':
                        recommendations.append("Your blood oxygen levels have been decreasing. Practice deep breathing exercises and consider consulting your doctor.")
                
                elif data_type == 'glucose':
                    if direction == 'increasing':
                        recommendations.append("Your glucose levels have been rising. Consider adjusting your diet and increasing physical activity.")
                    else:
                        recommendations.append("Your glucose levels have been decreasing. Ensure you're eating regular meals with balanced carbohydrates.")
        
        # Add general recommendations based on patterns
        daily_patterns = [p for p in results.get('patterns', []) if p['pattern_type'] == 'daily']
        for pattern in daily_patterns:
            data_type = pattern['data_type']
            peak_hour = pattern['peak_hour']
            
            if data_type == 'heart_rate' and peak_hour >= 22:
                recommendations.append("Your heart rate peaks late at night. Consider adjusting your evening routine to promote better sleep.")
        
        # Add general health recommendations if we don't have many specific ones
        if len(recommendations) < 2:
            recommendations.append("Maintain a regular sleep schedule and aim for 7-8 hours of quality sleep each night.")
            recommendations.append("Stay hydrated by drinking at least 8 glasses of water daily.")
            recommendations.append("Incorporate regular physical activity into your routine, aiming for at least 30 minutes of moderate exercise most days.")
        
        return recommendations

    def _render_template(self, template_path: Optional[str], context: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Render a notification template with the given context.
        
        Args:
            template_path: Path to the template file
            context: Context data for rendering
            
        Returns:
            Rendered template as string or dictionary
        """
        if not template_path or not os.path.exists(template_path):
            # Return a default template based on context
            if 'level' in context and context['level'] in ['ALERT', 'CRITICAL']:
                return f"URGENT: {context['subject']} - {context['message']}"
            else:
                return f"{context['subject']} - {context['message']}"
        
        try:
            # Check if it's a JSON template
            if template_path.endswith('.json'):
                with open(template_path, 'r') as f:
                    template_data = json.load(f)
                
                # Simple variable substitution in strings
                if isinstance(template_data, dict):
                    result = {}
                    for key, value in template_data.items():
                        if isinstance(value, str):
                            for var_name, var_value in context.items():
                                if isinstance(var_value, (str, int, float, bool)):
                                    value = value.replace(f"{{{var_name}}}", str(var_value))
                            result[key] = value
                        else:
                            result[key] = value
                    return result
                return template_data
            
            # Text or HTML template
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Simple variable substitution
            for var_name, var_value in context.items():
                if isinstance(var_value, (str, int, float, bool)):
                    template_content = template_content.replace(f"{{{var_name}}}", str(var_value))
            
            return template_content
        
        except Exception as e:
            self.logger.error(f"Failed to render template {template_path}: {str(e)}")
            return f"{context.get('subject', 'Notification')} - {context.get('message', '')}"
    
    def _log_notification(self, notification_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Log a notification.
        
        Args:
            notification_data: Notification data
            results: Results of sending the notification
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'notification': notification_data,
            'results': results
        }
        
        self.logger.info(f"Notification sent: {notification_data['level']} to {notification_data['user_id']} - {notification_data['subject']}")
        
        # In a real system, you might store this in a database
