import torch

batch = 200
inputs = (torch.rand(batch, 12) * 12).long()
query = (torch.rand(batch, 1)*12).long()
emb = torch.nn.Embedding(12, 512)
output_inputs = emb(inputs)
output_query = emb(query)