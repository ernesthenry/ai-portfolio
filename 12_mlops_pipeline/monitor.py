import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    """
    Monitors data distribution changes (Drift).
    Uses Kolmogorov-Smirnov (KS) test.
    """
    def __init__(self, reference_data):
        self.reference_data = reference_data # The training data
        
    def check_drift(self, new_batch, threshold=0.05):
        """
        Returns True if drift is detected (p_value < threshold).
        """
        print("Checking for Data Drift...")
        stat, p_value = ks_2samp(self.reference_data, new_batch)
        
        if p_value < threshold:
            print(f"🚨 DRIFT DETECTED! P-value: {p_value:.5f}")
            return True
        else:
            print(f"✅ Data stable. P-value: {p_value:.5f}")
            return False

if __name__ == "__main__":
    # Simulate Reference Data (Training: Mean=50)
    train_dist = np.random.normal(50, 10, 1000)
    monitor = DriftDetector(train_dist)
    
    # Batch 1: Same distribution
    print("\n--- Batch 1 (Same Dist) ---")
    batch_1 = np.random.normal(50, 10, 100)
    monitor.check_drift(batch_1)
    
    # Batch 2: Shifted distribution (Mean=60) -> Should trigger alarm
    print("\n--- Batch 2 (Shifted Mean) ---")
    batch_2 = np.random.normal(60, 10, 100)
    monitor.check_drift(batch_2)
