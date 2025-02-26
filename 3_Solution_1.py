"""
This stupid technique to manage the memories with just offload the KV Caches from the attentions. 
This reduces the overhead communication with improving the performance instead using KV Cache for all positions.

And actually we use this method to reconstruct the architecture of qwen2.5 32B coder
"""

import numpy as np

class KVCacheHead:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)  # Simulate the cache data
        self.is_offloaded = False

    def offload(self):
        self.is_offloaded = True

    def access(self, index):
        if self.is_offloaded:
            print("Retrieving data from offloaded memory (slow)...")
        return self.data[index]  # Fast or slow access depending on offload status

class KVCache:
    def __init__(self, num_heads, head_size, gpu_memory):
        self.num_heads = num_heads
        self.head_size = head_size
        self.gpu_memory = gpu_memory
        self.heads = [KVCacheHead(head_size) for _ in range(num_heads)]
        self.offload_strategy()

    def offload_strategy(self):
        total_memory = self.num_heads * self.head_size
        if total_memory <= self.gpu_memory:
            print("All heads fit in GPU.")
            return

        # Simulation: Offload heads with the highest index (assume these are less important)
        heads_to_offload = total_memory - self.gpu_memory  # Memory that needs to be freed
        heads_to_offload_count = int(heads_to_offload / self.head_size)

        for i in range(self.num_heads - 1, max(-1, self.num_heads - 1 - heads_to_offload_count), -1):
            print(f"Offloading head {i}...")
            self.heads[i].offload()

    def access(self, head_index, index):
        return self.heads[head_index].access(index)


batch_size = 32
num_layers = 12
sequence_length = 80000
hidden_dimension = 768
num_heads = 8  # Number of attention heads
head_size = 2 * batch_size * num_layers * sequence_length * (hidden_dimension // num_heads) * 4  # sizeof(float32)

gpu_memory = 8 * 1024 * 1024 * 1024  # 8GB

# Create KV Cache
kv_cache = KVCache(num_heads, head_size, gpu_memory)

# Access data
print("Accessing head 0, index 100:", kv_cache.access(0, 100))
print("Accessing head 5, index 100:", kv_cache.access(5, 100))  # Might be slow if offloaded
