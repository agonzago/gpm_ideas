# Add this standalone verification test to debug your Kalman filter implementation
import os
import time
import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import random

# Assuming KalmanFilter is in the same directory
from Kalman_filter_jax import KalmanFilter

def verify_kalman_filter():
    """
    Creates a simple linear state-space model and tests the KalmanFilter class.
    This verification is separate from the DSGE model to isolate Kalman filter issues.
    """
    print("\n=== KALMAN FILTER VERIFICATION TEST ===")
    
    # --- 1. Set up a simple state-space model ---
    n_state = 3
    n_obs = 2
    n_shocks = 2
    T = 100  # time periods
    
    # Create stable matrices (eigenvalues less than 1)
    key = random.PRNGKey(0)
    key1, key2, key3, key4, key5 = random.split(key, 5)
    
    # Create state transition (A in typical notation, T in KalmanFilter)
    A_raw = random.normal(key1, (n_state, n_state))
    Q, R = jnp.linalg.qr(A_raw)  # Orthogonal decomposition
    eigenvalues = jnp.diag(jnp.array([0.8, 0.5, 0.3]))  # Stable eigenvalues
    A = Q @ eigenvalues @ Q.T  # Should be stable
    
    # Create shock impact (B in typical notation, R in KalmanFilter)
    B = random.normal(key2, (n_state, n_shocks))
    
    # Create observation matrix (C in typical notation, C in KalmanFilter)
    C = random.normal(key3, (n_obs, n_state))
    
    # Create observation noise (D in typical notation, H in KalmanFilter)
    D_diag = jnp.exp(random.normal(key4, (n_obs,)) - 2.0)  # Small positive values
    D = jnp.diag(D_diag)
    
    # Initial state and covariance
    x0 = jnp.zeros(n_state)
    P0 = jnp.eye(n_state)
    
    print("Created test state-space model:")
    print(f"  State dimension: {n_state}")
    print(f"  Observation dimension: {n_obs}")
    print(f"  Shock dimension: {n_shocks}")
    print(f"  Time periods: {T}")
    
    # --- 2. Simulate data ---
    print("\nSimulating data...")
    
    def simulate_step(state, key_t):
        key_s, key_o = random.split(key_t)
        state_noise = random.normal(key_s, (n_shocks,))
        obs_noise = random.normal(key_o, (n_obs,))
        
        next_state = A @ state + B @ state_noise
        obs = C @ next_state + obs_noise * jnp.sqrt(D_diag)
        
        return next_state, (next_state, obs)
    
    keys = random.split(key5, T)
    _, (states, observations) = jax.lax.scan(simulate_step, x0, keys)
    
    print(f"Simulated data: states {states.shape}, observations {observations.shape}")
    
    # --- 3. Test KalmanFilter with the simulated data ---
    print("\nTesting KalmanFilter...")
    
    try:
        # Initialize KalmanFilter
        kf = KalmanFilter(
            T=A,       # State transition
            R=B,       # Shock impact 
            C=C,       # Observation matrix
            H=D,       # Observation noise covariance
            init_x=x0, # Initial state mean
            init_P=P0  # Initial state covariance
        )
        print("✓ KalmanFilter initialized successfully")
        
        # Test filter method
        try:
            filter_results = kf.filter(observations)
            print("✓ KalmanFilter.filter() successful")
            
            # Verify filter results
            x_filt = filter_results['x_filt']
            P_filt = filter_results['P_filt']
            
            print(f"  Filtered states shape: {x_filt.shape}")
            print(f"  Filtered covariance shape: {P_filt.shape}")
            
            # Calculate mean absolute error between filtered and true states
            mae = jnp.mean(jnp.abs(x_filt - states))
            print(f"  Mean Absolute Error between filtered and true states: {mae:.4f}")
            
            # Test log_likelihood method
            try:
                log_lik = kf.log_likelihood(observations)
                print(f"✓ KalmanFilter.log_likelihood() successful: {log_lik:.4f}")
                
                # Simple check if log_likelihood is reasonable
                if jnp.isfinite(log_lik) and log_lik < 0:
                    print("  Log-likelihood value looks reasonable")
                else:
                    print("  ⚠️ Warning: Log-likelihood value might be problematic")
                
                # --- 4. Test with NaN values ---
                print("\nTesting robustness to NaNs...")
                
                # Create data with a few NaN values
                obs_with_nans = observations.at[10:15, 0].set(jnp.nan)
                
                try:
                    # Check if KalmanFilter handles NaNs
                    ll_with_nans = kf.log_likelihood(obs_with_nans)
                    print(f"✓ KalmanFilter.log_likelihood() with NaNs: {ll_with_nans:.4f}")
                    
                    # Compare to original
                    print(f"  Difference: {log_lik - ll_with_nans:.4f}")
                    
                except Exception as e_nan:
                    print(f"✗ KalmanFilter does not handle NaNs well: {e_nan}")
                    print("  Note: If your KalmanFilter doesn't handle NaNs, this is expected")
                
            except Exception as e_ll:
                print(f"✗ Error in KalmanFilter.log_likelihood(): {e_ll}")
                raise
            
        except Exception as e_filter:
            print(f"✗ Error in KalmanFilter.filter(): {e_filter}")
            raise
        
    except Exception as e_init:
        print(f"✗ Error initializing KalmanFilter: {e_init}")
        raise
    
    print("\n=== VERIFICATION COMPLETE ===")
    return True

# Run the verification if this script is executed directly
if __name__ == "__main__":
    verify_kalman_filter()