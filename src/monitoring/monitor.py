from prometheus_client import start_http_server, Gauge, Counter
import time
import numpy as np
from datetime import datetime

PREDICTION_LATENCY = Gauge('model_prediction_latency_seconds', 
                          'Time for model prediction')
PREDICTIONS_TOTAL = Counter('model_predictions_total', 
                          'Total number of predictions')
PREDICTION_ERROR = Gauge('model_prediction_error', 
                        'Prediction error metrics')

class ModelMonitor:
    def __init__(self):
        start_http_server(8000)
        self.predictions = []
        self.latencies = []
        
    def record_prediction(self, prediction, actual=None, latency=None):
        self.predictions.append(prediction)
        PREDICTIONS_TOTAL.inc()
        
        if latency:
            PREDICTION_LATENCY.set(latency)
            self.latencies.append(latency)
            
        if actual is not None:
            error = abs(prediction - actual)
            PREDICTION_ERROR.set(error)