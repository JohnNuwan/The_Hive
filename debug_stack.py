
import numpy as np
import jax.numpy as jnp

# Simulate what we think is happening
try:
    # Scenario 1: Empty list
    print("Scenario 1: Stack empty list")
    try:
        np.stack([])
    except Exception as e:
        print(f"Caught: {e}")

    # Scenario 2: List of empty arrays
    print("\nScenario 2: Stack list of empty arrays")
    arr = np.array([])
    print(f"Empty array shape: {arr.shape}")
    stacked = np.stack([arr, arr])
    print(f"Stacked shape: {stacked.shape}")

    # Scenario 3: List of (0, 2560) arrays
    print("\nScenario 3: Stack (0, 2560) arrays")
    arr = np.zeros((0, 2560))
    stacked = np.stack([arr]*128)
    print(f"Stacked shape: {stacked.shape}")
    
    # Check JAX behavior
    print("\nJAX Array from Scenario 3")
    j_arr = jnp.array(stacked)
    print(f"JAX shape: {j_arr.shape}")

except Exception as e:
    print(e)
