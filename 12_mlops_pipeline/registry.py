import pickle
import os
import time
import json

class ModelRegistry:
    """
    Simulates MLflow or AWS SageMaker Registry.
    Tracks model versions, parameters, and metrics.
    """
    def __init__(self, registry_dir="./model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
    
    def log_model(self, model, name, metrics, params):
        timestamp = int(time.time())
        version = f"{name}_v{timestamp}"
        
        # 1. Save Artifact (The serialized model)
        model_path = os.path.join(self.registry_dir, f"{version}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        # 2. Save Metadata (The metrics)
        metadata = {
            "name": name,
            "version": version,
            "metrics": metrics,
            "parameters": params,
            "timestamp": timestamp
        }
        meta_path = os.path.join(self.registry_dir, f"{version}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Model registered: {version}")
        return version

if __name__ == "__main__":
    # Mock usage
    reg = ModelRegistry()
    reg.log_model(
        model={"mock": "sklearn_object"}, 
        name="churn_predictor", 
        metrics={"accuracy": 0.89}, 
        params={"max_depth": 5}
    )
