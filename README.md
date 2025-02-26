We use this technique to make more efficient KV Caching on our agent.
Not just fine tuning, but reconstruct the base architecture of the pretrained models, which is Qwen2.5 32B Coder Instruct.
We use the GSM, LongCOT, Science, OpenMath, and Coding Datasets.

Not fast, this is long training, we chunk the memories into extremely small batch, so it can fits into 4 x H100 GPUs, for 2 epochs, for 4 days training.

Result? It fast, eventhough just 93% from the original model, but it significantly fast, and can beats GPT-4o for daily tasks and small software development.



Special credit -> Hang Tou, Wanto, Jo. Dani, Yun Han Que
