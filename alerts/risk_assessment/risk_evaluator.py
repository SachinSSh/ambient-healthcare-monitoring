from typing import Dict, Any, List, Tuple
import yaml
import os

class RiskEvaluator:
    """Evaluates health risks based on anomaly detection results."""
    
    RISK_LEVELS = {
        'LOW': 1,
        'MODERATE': 2,
        'HIGH': 3,
        'CRITICAL': 4
    }
    
    def __init__(self, config_path: str = None):
        self.thresholds = self._load_thresholds(config_path)
    
    def _load_thresholds(self, config_path: str = None) -> Dict[str, Any]:
        """Load risk thresholds from configuration file."""
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'alert_thresholds.yaml'
            )
        
        if not os.path.exists(config_path):
            # Default thresholds if file doesn't exist
            return {
                'heart_rate': {
                    'CRITICAL': {'min': 40, 'max': 180},
                    'HIGH': {'min': 50, 'max': 160},
                    'MODERATE': {'min': 55, 'max': 140}
                },
                'blood_oxygen': {
                    'CRITICAL': {'min': 85},
                    'HIGH': {'min': 90},
                    'MODERATE': {'min': 94}
                },
                'temperature': {
                    'CRITICAL': {'min': 35, 'max': 39.5},
                    'HIGH': {'min': 35.5, 'max': 39},
                    'MODERATE': {'min': 36, 'max': 38.5}
                }
            }
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def evaluate_risk(self, data: Dict[str, Any], anomaly_results: Dict[str, Tuple[bool, float]]) -> Dict[str, Any]:
        """
        Evaluate health risk based on data and anomaly detection results.
        
        Args:
            data: The health data
            anomaly_results: Dictionary mapping data types to (is_anomaly, anomaly_score) tuples
        
        Returns:
            Dict containing risk assessment results
        """
        risk_assessment = {
            'overall_risk_level': 'LOW',
            'overall_risk_score': 0,
            'risk_factors': [],
            'recommendations': []
        }
        
        # Evaluate risk for each vital sign
        risk_factors = []
        max_risk_level = 'LOW'
        max_risk_score = 0
        
        if 'heart_rate' in data['data'] and 'heart_rate' in anomaly_results:
            hr_risk = self._evaluate_heart_rate_risk(data['data']['heart_rate'], anomaly_results['heart_rate'])
            if hr_risk['risk_level'] != 'LOW':
                risk_factors.append(hr_risk)
                if self.RISK_LEVELS[hr_risk['risk_level']] > self.RISK_LEVELS[max_risk_level]:
                    max_risk_level = hr_risk['risk_level']
                    max_risk_score = max(max_risk_score, hr_risk['risk_score'])
        
        if 'blood_oxygen' in data['data'] and 'blood_oxygen' in anomaly_results:
            spo2_risk = self._evaluate_blood_oxygen_risk(data['data']['blood_oxygen'], anomaly_results['blood_oxygen'])
            if spo2_risk['risk_level'] != 'LOW':
                risk_factors.append(spo2_risk)
                if self.RISK_LEVELS[spo2_risk['risk_level']] > self.RISK_LEVELS[max_risk_level]:
                    max_risk_level = spo2_risk['risk_level']
                    max_risk_score = max(max_risk_score, spo2_risk['risk_score'])
        
        # Update overall risk assessment
        risk_assessment['overall_risk_level'] = max_risk_level
        risk_assessment['overall_risk_score'] = max_risk_score
        risk_assessment['risk_factors'] = risk_factors
        
        # Generate recommendations based on risk factors
        risk_assessment['recommendations'] = self._generate_recommendations(risk_factors)
        
        return risk_assessment
    
    def _evaluate_heart_rate_risk(self, hr_data: Any, anomaly_result: Tuple[bool, float]) -> Dict[str, Any]:
        """Evaluate risk level for heart rate data."""
        is_anomaly, anomaly_score = anomaly_result
        
        # Extract heart rate value
        if isinstance(hr_data, list) and len(hr_data) > 0 and 'value' in hr_data[0]:
            hr_value = hr_data[0]['value']
        else:
            hr_value = hr_data.get('value', 0)
        
        # Determine risk level based on thresholds
        risk_level = 'LOW'
        for level in ['CRITICAL', 'HIGH', 'MODERATE']:
            thresholds = self.thresholds.get('heart_rate', {}).get(level, {})
            min_val = thresholds.get('min', 0)
            max_val = thresholds.get('max', 300)
            
            if hr_value < min_val or hr_value > max_val:
                risk_level = level
                break
        
        # If anomaly detection flagged it but thresholds didn't, set to MODERATE
        if is_anomaly and risk_level == 'LOW':
            risk_level = 'MODERATE'
        
        return {
            'vital_sign': 'heart_rate',
            'value': hr_value,
            'risk_level': risk_level,
            'risk_score': self.RISK_LEVELS[risk_level] * anomaly_score,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score
        }
    
    def _evaluate_blood_oxygen_risk(self, spo2_data: Any, anomaly_result: Tuple[bool, float]) -> Dict[str, Any]:
        """Evaluate risk level for blood oxygen data."""
        is_anomaly, anomaly_score = anomaly_result
        
        # Extract SpO2 value
        if isinstance(spo2_data, list) and len(spo2_data) > 0 and 'value' in spo2_data[0]:
            spo2_value = spo2_data[0]['value']
        else:
            spo2_value = spo2_data.get('value', 0)
        
        # Determine risk level based on thresholds
        risk_level = 'LOW'
        for level in ['CRITICAL', 'HIGH', 'MODERATE']:
            thresholds = self.thresholds.get('blood_oxygen', {}).get(level, {})
            min_val = thresholds.get('min', 0)
            
            if spo2_value < min_val:
                risk_level = level
                break
        
        # If anomaly detection flagged it but thresholds didn't, set to MODERATE
        if is_anomaly and risk_level == 'LOW':
            risk_level = 'MODERATE'
        
        return {
            'vital_sign': 'blood_oxygen',
            'value': spo2_value,
            'risk_level': risk_level,
            'risk_score': self.RISK_LEVELS[risk_level] * anomaly_score,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score
        }
    
    def _generate_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        
        for factor in risk_factors:
            vital_sign = factor['vital_sign']
            risk_level = factor['risk_level']
            value = factor['value']
            
            if vital_sign == 'heart_rate':
                if value > 100 and risk_level in ['MODERATE', 'HIGH', 'CRITICAL']:
                    recommendations.append("Your heart rate is elevated. Consider resting and practicing deep breathing.")
                    if risk_level in ['HIGH', 'CRITICAL']:
                        recommendations.append("If heart rate remains elevated, consult a healthcare provider.")
                elif value < 60 and risk_level in ['MODERATE', 'HIGH', 'CRITICAL']:
                    recommendations.append("Your heart rate is lower than normal. Monitor for dizziness or fatigue.")
                    if risk_level in ['HIGH', 'CRITICAL']:
                        recommendations.append("If you experience dizziness or weakness, consult a healthcare provider.")
            
            elif vital_sign == 'blood_oxygen':
                if risk_level in ['MODERATE']:
                    recommendations.append("Your blood oxygen level is slightly below normal. Try deep breathing exercises.")
                elif risk_level in ['HIGH', 'CRITICAL']:
                    recommendations.append("Your blood oxygen level is concerning. If you experience shortness of breath, seek medical attention.")
        
        # Add general recommendations if any risk factors exist
        if risk_factors:
            recommendations.append("Continue monitoring your vital signs closely.")
        

        return recommendations
