import torch

class SimpleKVStore:
    def __init__(self, embedding_dim):
        self.keys = []
        self.values = []
        self.embedding_dim = embedding_dim

    def add(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def retrieve(self, query):
        if not self.keys:
            return None  # Null Cache

        # Calculate the similarity using dot product
        similarities = [torch.dot(query, key) for key in self.keys]

        # Find the highest score
        best_index = similarities.index(max(similarities))

        # Return the best scores
        return self.values[best_index]

embedding_dim = 10  # Dimension
kv_store = SimpleKVStore(embedding_dim)

# Example e.g. of words' vectors
kata_pizza = torch.randn(embedding_dim)  
kata_pasta = torch.randn(embedding_dim) 
kata_italia = torch.randn(embedding_dim)  

# Store the cache
kv_store.add(kata_pizza, torch.tensor([0.9]))  
kv_store.add(kata_pasta, torch.tensor([0.8]))
kv_store.add(kata_italia, torch.tensor([0.7]))

# Now query
query = torch.randn(embedding_dim)  

# Find the relevant queries
retrieved_value = kv_store.retrieve(query)

if retrieved_value is not None:
    print("Most relevant keys", retrieved_value)
else:
    print("Null :()")


# Extremely simple simulation
jumlah_entri = len(kv_store.keys)
ukuran_key_per_entri = embedding_dim * 4  # Float32
ukuran_value_per_entri = 4 # Float32

total_memory = jumlah_entri * (ukuran_key_per_entri + ukuran_value_per_entri)
print(f"Total memory: {total_memory}") # The value you watch just in one token. How if it has 1 million context length? Your GPU fucked up
