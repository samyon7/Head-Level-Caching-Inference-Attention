import numpy as np

def calculate_kv_cache_size(batch_size, sequence_length, hidden_dimension, num_layers, num_heads, sizeof_datatype):
    return 2 * batch_size * num_layers * sequence_length * hidden_dimension * sizeof_datatype

def calculate_kv_cache_size_headwise(batch_size, sequence_length, hidden_dimension, num_layers, num_heads, sizeof_datatype):
    dh = hidden_dimension // num_heads
    return num_layers * 2 * batch_size * sequence_length * dh * sizeof_datatype

# Simulation Parameters
batch_size = 32
sequence_length = 80000
hidden_dimension = 768
num_layers = 12
num_heads = 8
sizeof_datatype = 4  # float32 (4 bytes)

# Calculate KV Cache size using the standard approach
kv_cache_size_full = calculate_kv_cache_size(batch_size, sequence_length, hidden_dimension, num_layers, num_heads, sizeof_datatype)

# Calculate KV Cache size using HEADINFER
kv_cache_size_headwise = calculate_kv_cache_size_headwise(batch_size, sequence_length, hidden_dimension, num_layers, num_heads, sizeof_datatype)

# Calculate the fraction (Alpha)
alpha = kv_cache_size_headwise / kv_cache_size_full

print(f"KV Cache Size (Standard Approach): {kv_cache_size_full / (1024 * 1024):.2f} MB")
print(f"KV Cache Size (HEADINFER): {kv_cache_size_headwise / (1024 * 1024):.2f} MB")
print(f"Fraction (Alpha): {alpha:.4f}")

# Verify the Alpha formula
alpha_verification = 1 / (num_layers * num_heads)
print(f"Alpha Verification (1 / (L * H)): {alpha_verification:.4f}")
