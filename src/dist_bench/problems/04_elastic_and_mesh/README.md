# Elastic Launchs and Fault Tolerance

Just read through the elastic docs, and this seems like something good to add to my learning.

I want to cover:
- Node departure (scale down)
- Node arrival (scale up)
- Handling worker failures

Also I might cover NUMA binings, the docs say this is "improve performance by binding worker processes to CPUs near their assigned GPUs":
- But I think this will be redundant on my home setup.
- I don't think I have easy experimental options with multiple CPU and GPU nodes outside of AWS.

