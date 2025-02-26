"""
Aight, so this code snippet's gonna spill the tea on how the KV Cache gets divvied up – some on the GPU, some offloaded, you feel me? 
Like, think of it as a digital asset allocation, but for your model's memory. 
You can tweak the knobs – GPU memory, sequence length, the whole shebang – and peep how it messes with the offloading shuffle. 
It's basically a high-level, conceptual illustration, not a full-blown, production-grade, optimized memory manager. 
We're talking about moving chunks of NumPy arrays around, not diving deep into the silicon and wrangling memory pages, dig? 
So yeah, mess around with the parameters and see how this digital ballet plays out. It's educational, yo, but remember, real-world implementations? 

Way more intricate, fam.
"""


import numpy as np

class KVCache:
    def __init__(self, total_size, gpu_memory):
        self.total_size = total_size
        self.gpu_memory = gpu_memory
        self.on_gpu = None  # Array for data on GPU
        self.offloaded = None  # Array for offloaded data
        self.alpha = 0.0  # Fraction of data on the GPU

        self.offload()  # Initialize with offloading

    def offload(self):
        if self.total_size <= self.gpu_memory:
            self.alpha = 1.0
            self.on_gpu = np.zeros(self.total_size, dtype=np.float32)  
            self.offloaded = None
            print("All KV Cache is on the GPU.")
        else:
            self.alpha = self.gpu_memory / self.total_size
            gpu_size = int(self.alpha * self.total_size)
            offload_size = self.total_size - gpu_size

            self.on_gpu = np.zeros(gpu_size, dtype=np.float32)  # Simulate memory on GPU
            self.offloaded = np.zeros(offload_size, dtype=np.float32)  # Simulate memory on CPU/Disk
            print(f"Partial KV Cache is offloaded. Alpha: {self.alpha:.2f}")

    def access(self, index):
        """Access data at a specific index."""
        if index < len(self.on_gpu):
            return self.on_gpu[index]  
        else:
            offload_index = index - len(self.on_gpu)
            print("Fetching data from offloaded memory (slow)...")
            return self.offloaded[offload_index]  
          
batch_size = 32
num_layers = 12
sequence_length = 2048  
hidden_dimension = 768
sizeof_datatype = 4 

total_kv_size = 2 * batch_size * num_layers * sequence_length * hidden_dimension * sizeof_datatype

# GPU Memory Size (HBM)
gpu_memory = 8 * 1024 * 1024 * 1024  # For example, 8GB

# Create KV Cache
kv_cache = KVCache(total_kv_size, gpu_memory)

# Access data
print("Accessing data at index 100:", kv_cache.access(100))  # Fast access
print("Accessing data at a distant index:", kv_cache.access(kv_cache.total_size - 1))  # May require offload
