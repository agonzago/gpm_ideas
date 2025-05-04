import jax

# Get the number of CPU devices
cpu_count = len(jax.devices("cpu"))
print(f"Number of CPU devices: {cpu_count}")

# Get the number of GPU devices
gpu_count = len(jax.devices("gpu"))
print(f"Number of GPU devices: {gpu_count}")

# Get the total number of all available devices
all_devices_count = len(jax.devices())
print(f"Total number of available devices: {all_devices_count}")


import tensorflow_probability.substrates.jax as tfp
import jax
import sys

print(f"--- TFP Exploration ---")
print(f"Python version: {sys.version}")
print(f"JAX version: {jax.__version__}")

try:
    import tensorflow_probability as tfp_root
    print(f"TFP version: {tfp_root.__version__}")
except Exception as e:
    print(f"Could not get TFP version: {e}")

# Check standard path
print(f"\nChecking tfp.linalg...")
try:
    linalg_module = tfp.linalg
    print(f"  tfp.linalg found.")
    has_op = hasattr(linalg_module, 'LinearOperatorFullMatrix')
    print(f"  tfp.linalg has LinearOperatorFullMatrix: {has_op}")
    if not has_op:
         print(f"  Attributes in tfp.linalg: {dir(linalg_module)}")
except AttributeError:
    print(f"  tfp.linalg not found directly.")
except Exception as e:
    print(f"  Error accessing tfp.linalg: {e}")

# Check experimental path (based on your previous fix)
print(f"\nChecking tfp.experimental.linalg...")
try:
    experimental_module = tfp.experimental
    print(f"  tfp.experimental found.")
    try:
        exp_linalg_module = experimental_module.linalg
        print(f"  tfp.experimental.linalg found.")
        has_op_exp = hasattr(exp_linalg_module, 'LinearOperatorFullMatrix')
        print(f"  tfp.experimental.linalg has LinearOperatorFullMatrix: {has_op_exp}")
        if not has_op_exp:
            print(f"  Attributes in tfp.experimental.linalg: {dir(exp_linalg_module)}")
    except AttributeError:
        print(f"  tfp.experimental does not have 'linalg'.")
    except Exception as e:
        print(f"  Error accessing tfp.experimental.linalg: {e}")
except AttributeError:
    print(f"  tfp.experimental not found.")
except Exception as e:
    print(f"  Error accessing tfp.experimental: {e}")

print(f"\n--- End Exploration ---")