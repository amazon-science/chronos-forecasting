from src.chronos.chronos2.pipeline import Chronos2Pipeline

import torch
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
)
random_data = torch.rand(120, 6, 60).to("cuda")
print("*****************************************")
print(pipeline.embed(random_data)[0][0].shape)



random_data = torch.rand(20,200,100).to("cuda")
# print(f" the embedding shape of input shape {random_data.shape} is : {pipeline.embed(random_data).shape}")
embeds = pipeline.embed(random_data)[0]

print(type(embeds))
print(len(embeds))

for i, x in enumerate(embeds):
    print(f"Element {i}:")
    print("  type :", type(x))
    print("  shape:", x.shape)
    print("  dtype:", x.dtype)
    print("  device:", x.device)