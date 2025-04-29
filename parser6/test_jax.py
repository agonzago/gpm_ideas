import jax
import jax.numpy as jnp
import sys

print("Python:", sys.version)
print("JAX version:", jax.__version__) # Will still be 0.6.0

# Define function WITHOUT decorator
def foo_impl(x, n):
    return x + jnp.ones(n)

# Apply jit AFTER definition
foo = jax.jit(foo_impl, static_argnames=('n',))

try:
    result = foo(jnp.array(1.0), 5)
    print("Minimal example (alternative syntax) worked:", result)
except Exception as e:
    print("Minimal example (alternative syntax) FAILED:", e)
    import traceback
    traceback.print_exc()