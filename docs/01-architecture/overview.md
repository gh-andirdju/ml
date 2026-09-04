# Proposed system architecture

## Recommendation

Use a hybrid design with one portable Python codebase and platform-specific
runtime layers:

```mermaid
flowchart LR
    subgraph shared[Shared application concepts]
        model[PyTorch Geometric<br/>model code]
        adapter[Neo4j-to-tensor<br/>data adapter]
        checkpoint[Device-neutral<br/>checkpoints]
    end

    subgraph mac[Mac development]
        macpy[Host Python]
        mps[MPS or CPU]
        localneo[Neo4j<br/>proposed Apple Container]
        volume[(Persistent volume)]
        macpy --> mps
        localneo --> volume
    end

    subgraph dc[Self-managed data center]
        subgraph cluster[Linux H200 cluster]
            trainer[Versioned training container]
            cuda[CUDA on H200 nodes]
            trainer --> cuda
        end
        subgraph dbtier[Separate Linux database tier]
            dcneo[Version-pinned Neo4j]
            storage[(Persistent database storage)]
            dcneo --> storage
        end
    end

    adapter <--> localneo
    adapter <--> dcneo
    model --> macpy
    model --> trainer
    checkpoint <--> macpy
    checkpoint <--> trainer
```

The Apple Container choice for local Neo4j is still proposed, not accepted.
Native Homebrew Neo4j remains a simpler fallback.

The production target is confirmed as a single self-managed data center. The
H200 compute cluster and Neo4j database tier both run on Linux and communicate
over the internal data-center network. Neo4j releases are pinned explicitly.

## Boundaries

- Neo4j owns durable graph data, indexes, constraints, and graph queries.
- The Python data layer converts query results into CPU tensors.
- PyTorch Geometric owns sampling, message passing, training, and inference.
- A device adapter moves tensors and models to MPS, CUDA, or CPU.
- CUDA-only optimization belongs behind an optional server-specific boundary.

```mermaid
flowchart LR
    neo4j[(Neo4j)] -->|Cypher over Bolt| ingestion[Python ingestion]
    ingestion -->|CPU tensors| pyg[PyTorch Geometric]
    pyg --> device{Runtime device}
    device --> mps[MPS on Mac]
    device --> cuda[CUDA on H200]
    device --> cpu[CPU fallback]
    cuda --> optional[Optional CUDA-only optimization]
```

## Why not put everything in Apple Container?

The Mac's Apple Container runtime does not currently provide GPU passthrough to
Linux containers. PyTorch inside it would therefore lose access to MPS and train
on CPU. Host-native Python is the better local training environment.

## Why not run Neo4j on the H200 GPU nodes?

Neo4j is primarily a CPU, memory, and storage workload. The H200 should be
reserved for tensor computation. Production places the self-managed Neo4j
service on a separate Linux database tier in the same data center.
