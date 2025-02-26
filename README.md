We use this technique to make more efficient KV Caching on our agent.
Not just fine tuning, but reconstruct the base architecture of the instruct models, which is Qwen2.5 32B Coder Instruct, which seems like a pretrained model. Not from .bin file, but construct the pytorch model, replicate the head structure, replicate the technique, and load it into transformer model.
We use the GSM, LongCOT, Science, OpenMath, Instruct Question Answering, and Coding Datasets.
We use Supervised Fine Tuning, and for future, we will use some *heuristics* method. Just wait, we will use the spike neural networks, and embedd it into this model agent.

Not fast, this is long training, we chunk the memories into extremely small batch, so it can fits into 4 x H100 GPUs, for 2 epochs, for 4 days training.

Result? It fast, eventhough just 93% from the original model, but it significantly fast, and can beats GPT-4o for daily tasks and small software development.



Special credit -> Hang Tou, Wanto, Jo. Dani, Yun Han Que
