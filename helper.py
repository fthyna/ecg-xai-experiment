import time
import torch
import os

def benchmark(func, *args, runs=10, **kwargs):
	"""
	Helper function to measure time taken to execute a function
	"""
	times = []
	results = []
	for _ in range(runs):
		torch.cuda.synchronize()
		start = time.perf_counter()
		result = func(*args, **kwargs)
		torch.cuda.synchronize()
		end = time.perf_counter()
		times.append(end - start)
		results.append(result)
	avg_time = sum(times) / runs
	print(f"{func.__name__} avg time over {runs} runs: {avg_time:.6f} seconds")
	return times, results

def transform_input(x):
	"""
	Transforms a (N, C) or (B, N, C) array into a (B, C, N) Tensor.
	Used to transform raw signal data into model-ready input with appropriate dimensions.
	Args:
		x (np.ndarray): (N, C) or (B, N, C) array.
	Returns:
		torch.Tensor: (B, C, N) tensor.
	"""
	xt = torch.tensor(x, dtype=torch.float32)
	if len(xt.shape) == 2:
		xt = xt.unsqueeze(0)
	return xt.permute(0, 2, 1)

def print_memory_usage(model, input_tensor):
	torch.cuda.empty_cache()
	# Dummy forward pass to see memory usage
	with torch.no_grad():
		model(input_tensor)
	allocated = torch.cuda.memory_allocated()
	reserved = torch.cuda.memory_reserved()
	print(f"Allocated: {allocated / 1024 ** 2:.2f} MB")
	print(f"Reserved: {reserved / 1024 ** 2:.2f} MB")

def check_create_folder(fldr):
	'''fldr: new folder path'''
	os.makedirs(fldr, exist_ok=True)