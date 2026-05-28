import os
import sys
import time
import jax
import jax.numpy as jnp
import optax

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src", "eva-lab"))

from eva_lab.muzero.jax_trainer import MuZeroTrainerJAX

class MockConfig:
    learning_rate = 1e-3
    weight_decay = 1e-4

class MockTransformedNets:
    apply = (lambda *args: None, lambda *args: None)
    
    def init(self, key, sample_obs):
        return {}

def test_jit_property_caching():
    print(f"JAX Devices: {jax.devices()}")
    
    # Instantiate trainer
    config = MockConfig()
    nets = MockTransformedNets()
    trainer = MuZeroTrainerJAX(config, nets)
    
    # 1. Access update_fn property multiple times and check identity
    print("Checking update_fn property identity...")
    fn1 = trainer.update_fn
    fn2 = trainer.update_fn
    
    is_same = (fn1 is fn2)
    print(f"Are fn1 and fn2 the exact same object? {is_same}")
    
    if is_same:
        print("SUCCESS: JIT compiled function is properly cached! No recompilation will happen on property access.")
    else:
        print("FAILURE: JIT function is NOT cached. Property access recreates it every time, forcing JAX recompilations.")
        
    assert is_same, "Identity check failed!"

if __name__ == "__main__":
    test_jit_property_caching()
