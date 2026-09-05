# Architecture summary

## Current direction

Use one portable PyTorch/PyG codebase with platform-specific runtime adapters.
Develop on macOS with MPS or CPU and run production workloads on CUDA in the
self-managed Linux H200 cluster. Keep Neo4j on a separate database tier.

```mermaid
flowchart LR
    subgraph local[Apple Silicon development]
        macpy[Host Python]
        mps[MPS or CPU]
        localdb[Neo4j Community<br/>Apple Container]
        macpy --> mps
    end

    subgraph dc[Self-managed data center]
        h200[Linux H200 cluster<br/>CUDA workloads]
        proddb[Linux Neo4j tier<br/>version pinned]
        storage[(On-premises storage<br/>and backups)]
        proddb --> storage
    end

    shared[Portable PyTorch and PyG<br/>models, tensors, checkpoints]
    adapter[Neo4j to tensor adapter]

    shared --> macpy
    shared --> h200
    localdb <--> adapter
    proddb <--> adapter
    adapter --> shared
```

## Boundaries

- Neo4j owns durable graph data, indexes, constraints, and Cypher queries.
- Python converts query results into CPU tensors.
- PyTorch Geometric owns sampling, message passing, training, and inference.
- A device adapter selects CUDA, MPS, or CPU.
- CUDA-only kernels, NCCL, and H200 optimizations remain optional.

## Validation ladder

| POC | Compute | Data and database boundary |
| --- | --- | --- |
| 1 | Laptop MPS or CPU | Karate round trip through local Neo4j |
| 2 | Laptop MPS or CPU | Pinned WikiCS round trip through local Neo4j |
| 3 | Kaggle Tesla T4 CUDA | Karate export and local Neo4j import verified |
| 4 | Kaggle Tesla T4 CUDA | WikiCS export and local Neo4j import verified |
| 5 | Kaggle CPU only | Karate: 100% class agreement with POC 3 |
| 6 | Kaggle CPU only | WikiCS: 97.75% class agreement with POC 4 |
| 7 | Kaggle CPU only | Flickr GraphSAGE: 246.06 seconds, 42.81% test accuracy |
| 8 | Kaggle Tesla T4 CUDA | Flickr GraphSAGE: 6.31 seconds, 42.35% test accuracy |
| 9 | Kaggle CPU only | Wide Flickr GraphSAGE: 599.42 seconds, 42.33% test accuracy |
| 10 | Kaggle Tesla T4 CUDA | Wide Flickr: 17.17 seconds, 6.88 GB peak allocated |

All ten keep compute selection outside model definitions. Kaggle receives no
database credential and runs no Neo4j process.

## Open decisions

- Production Python and framework versions.
- H200 scheduler, storage, network, and node topology.
- Neo4j edition and production clustering model.
- Local NVIDIA workstation purchase.
