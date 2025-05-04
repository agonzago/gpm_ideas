import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
n_obs = 4
num_steps = 200
mean = jnp.zeros(n_obs, dtype=jnp.float64)
# Use the same H_obs_sim from your main script
cov = jnp.diag(jnp.array([0.05**2, 0.05**2, 0.01**2, 0.05**2], dtype=jnp.float64))
_MACHINE_EPSILON = jnp.finfo(jnp.float64).eps
cov_reg = cov + _MACHINE_EPSILON * jnp.eye(n_obs, dtype=jnp.float64)

@jax.jit
def generate_noise(k, m, c):
    print("Generating noise (inside JIT)...")
    return random.multivariate_normal(k, m, c, shape=(num_steps,), dtype=jnp.float64)

try:
    print(f"Using backend: {jax.default_backend()}")
    noise = generate_noise(key, mean, cov_reg)
    print("Noise generated successfully:", noise.shape)
    print(noise[:5])
except Exception as e:
    print(f"Error during standalone multivariate_normal test: {e}")
    import traceback
    traceback.print_exc()