import jax
try:
    print(f"BACKEND: {jax.default_backend()}")
    print(f"DEVICES: {jax.devices()}")
except Exception as e:
    print(f"ERROR: {e}")
