# Flickr 2,048-channel benchmark

- Status: Ready for execution
- POC 11: Flickr GraphSAGE on Kaggle CPU only
- POC 12: the identical workload on one Kaggle Tesla T4
- Third environment: host-native Apple MPS with CPU fallback disabled
- Scope: comparison only; no Neo4j import

This workload retains the complete open Flickr graph and increases each hidden
GraphSAGE layer to 2,048 channels. All three environments use FP32, the same
model, data split, seed, optimizer, requested 20 epochs, and early-stopping
patience of six.

```mermaid
flowchart LR
    data["Flickr: 89,250 nodes"] --> model["GraphSAGE: 2,048 hidden channels"]
    model --> mps["Apple MPS"]
    model --> cpu["Kaggle CPU only"]
    model --> cuda["One Kaggle Tesla T4"]
    mps --> proof["Three pairwise agreement checks"]
    cpu --> proof
    cuda --> proof
```

## Acceptance checks

- Every artifact passes schema and detached SHA-256 validation.
- Model and dataset metadata match exactly across MPS, CPU, and CUDA.
- All three predicted-class pairs agree on at least 95% of nodes.
- The CPU runner proves CUDA unavailable and records Linux `wait4` resource use.
- The CUDA runner proves single-T4 execution and at least 8 GiB peak allocation.
- The MPS runner proves MPS availability with CPU fallback disabled.
- Each Kaggle wrapper is pinned to the executable source revision it runs.

The final comparison will be recorded under `results/` only after all checks
pass. These prediction artifacts are not eligible for Neo4j import.
