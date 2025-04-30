import jax
import jax.numpy as jnp # Often good practice to import jax.numpy as jnp

# Check the installed JAX version
print(f"JAX version: {jax.__version__}")

# Check available devices
print(f"Available devices: {jax.devices()}")

# Check the default backend
try:
    print(f"Default backend: {jax.default_backend()}")
except Exception as e:
    print(f"Could not get default backend: {e}")

# Simple computation to test GPU usage
try:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    # Ensure the computation runs on the GPU
    # jax.device_put transfers data to the default device (GPU in this case)
    # Although JAX often does this automatically for operations.
    x_on_gpu = jax.device_put(x) 
    y = jnp.dot(x_on_gpu, x_on_gpu).block_until_ready() # Perform computation and wait

    # --- CORRECTED LINE ---
    # Access the .device attribute directly, without parentheses
    print(f"Simple JAX computation successful on backend: {y.device}")
    # You can also check the devices set for the array
    print(f"Array devices: {y.devices()}")

except Exception as e:
    print(f"JAX computation failed: {e}")
    print("Please double-check the code or report the error if unsure.")