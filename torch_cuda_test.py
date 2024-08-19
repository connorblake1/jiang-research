import torch
print(f"CUDA Available?: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
print(f"CUDA Device Active: {torch.cuda.get_device_name(torch.cuda.current_device())}")

