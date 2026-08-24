## Specs
Use fastfetch if you're on arch, its amazing
```
OS: Arch Linux x86_64
Host: MS-7D70 (1.0)
Kernel: Linux 7.1.4-arch1-1
CPU: AMD Ryzen 9 7900X3D (24) @ 5.66 GHz
GPU 1: NVIDIA GeForce RTX 5060 Ti [Discrete]
GPU 2: NVIDIA GeForce RTX 3090 [Discrete]
```
- NOTE: PCIE Bifrucation on my particular motherboard with both GPUs in.
- Both GPUs run at `x8`
- Even my 3090 single results are running on `x8` speeds 


# 01 Single GPU Benchmarking
This is to get me thinking hypothetically about how parts of ML workloads may speed up, bu approximately what amount, when moving to distributed settings

My goal here is to make guesses, and then benchmark for myself, and try to reason through why I might be surprised or was right.


## Model and data details
- A two layer MLP with 2 `ReLUs` serving batches from a dummy dataset
- MSE Loss
```
- input_dim = 1024
- hidden_dim = 2048
- output_dim = 10
- dataset_size = 10000
- batch_size = 128
```


##  01 - 3090 only
- GPU: `3090` (at `x8`)
- Batches:  `79`
- Bsz: `128`
- Params: `6_316_042`

79 batches:
- 1st batch (outlier):
  - `159.738ms`
- Final batch (also outlier)
  - `5.518ms`
- Across other 77 batches:
  - Avg: `0.829ms`
  - StD: `0.226ms`


## 02 - 5060ti only
- GPU: `3090` (at `x8`)
- Batches:  `79`
- Bsz: `128`
- Params: `6_316_042`

79 batches:
- 1st batch (outlier):
  - `252.731ms`
- Final batch (also outlier)
  - `5.498ms`
- Across other 77 batches:
  - Avg: `1.001ms`
  - StD: `0.033ms`

## Obsevations:
- Obviously the 3090 is faster on average, on startup (`JIT?`) in particular. But the std of its batches is higher. Interesting

## DDP Predictions:
If I used DDP to split this same total workload. What might I expect for:
- The amount of computation performed by each GPU?
- The amount of data processed by each GPU
- The time spent doing computation
- The amount of communication
- Total throughput
- Time per optimisation step

This all depends how it works under the hood. I've realised I'm not ready to answer this yet. I would assume we split batches, but then how exactly will gradient updates be done? By sharing gradients in a blocking way with the master rank process? Probably something more involved nowadays. Moving on.


# 02 `torchrun`
Ok, time to use torchrun
Across my 2 GPUs:
- `torchrun --standalone --nproc-per-node=2 torchrun_minimal.py`
I made a helper statement to print the rank and local rank of processes

Obviously with nothing else strapped in, this just trains the same model twice across different GPUs.

## 03 `DDP` Proper

