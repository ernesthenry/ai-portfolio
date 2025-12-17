import numpy as np
from scipy import stats

# BUSINESS PROBLEM: A/B Testing
# "Does the new Checkout Button Color (Variant B) increase conversions vs the old one (Control A)?"

def run_ab_test(conversions_A, total_A, conversions_B, total_B):
    print(f"--- A/B Test Results ---")
    rate_A = conversions_A / total_A
    rate_B = conversions_B / total_B
    print(f"Control (A) Rate: {rate_A:.2%}")
    print(f"Variant (B) Rate: {rate_B:.2%}")
    
    # 1. Check Significance (Z-Test for Proportions)
    # Null Hypothesis: The rates are the same.
    p_hat = (conversions_A + conversions_B) / (total_A + total_B)
    se = np.sqrt(p_hat * (1 - p_hat) * (1/total_A + 1/total_B))
    
    z_score = (rate_B - rate_A) / se
    p_value = stats.norm.sf(abs(z_score))*2  # Two-tailed test
    
    print(f"P-Value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("✅ Result is Statistically Significant (Reject Null Hypothesis)")
        if rate_B > rate_A:
            print("🚀 DEPLOY VARIANT B (It is better)")
        else:
            print("🛑 KEEP CONTROL A (B is worse)")
    else:
        print("🤷 Inconclusive. Requires more data.")

if __name__ == "__main__":
    # Simulation: We ran the experiment for 1 week
    run_ab_test(
        conversions_A=200, total_A=1000, # 20% conversion
        conversions_B=250, total_B=1000  # 25% conversion
    )
