# GNN and Neo4j environment

## Current summary

The smallest local workflow is implemented and verified on Apple Silicon:
host-native PyTorch/PyG trains a one-layer GCN on MPS while Neo4j Community runs
as a version-pinned Linux ARM64 Apple Container. Production remains a design for
a self-managed Linux H200 cluster and a separate Neo4j tier in one data center.

```mermaid
flowchart LR
    mac[Mac<br/>MPS or CPU] --> shared[Portable PyTorch and PyG]
    shared --> h200[Linux H200 cluster<br/>CUDA]
    localdb[Neo4j Community<br/>Apple Container] <--> shared
    graphdb[Version-pinned Neo4j<br/>Linux data center] <--> shared
```

The verified POC uses the bundled 34-node Karate Club dataset, a single
`GCNConv(34, 4)`, and a Neo4j round trip. Run it with:

```bash
bun run poc:test
bun run poc:verify
```

The first command replaces only the tagged POC subgraph, trains on MPS with CPU
fallback disabled, and writes predictions. The second command is read-only.
Regenerate all project PDFs with `bun run md:pdf:all`; output is written under
`pdf/` with the Markdown directory structure preserved.

## Documents

| Core | Decisions and validation |
| --- | --- |
| [Current laptop state](docs/00-status/current-state.md) | [Compute portability](docs/04-decisions/ADR-001-compute-portability.md) |
| [Architecture summary](docs/01-architecture/overview.md) | [Local Neo4j runtime](docs/04-decisions/ADR-002-neo4j-runtime.md) |
| [Apple Silicon development](docs/02-local-development/apple-silicon.md) | [Production topology](docs/04-decisions/ADR-003-production-topology.md) |
| [H200 cluster design](docs/03-deployment/h200-server.md) | [Local NVIDIA device](docs/04-decisions/ADR-004-local-nvidia-device.md) |
| [Implementation phases](docs/05-plan/phases.md) | [Minimal local proof](docs/06-validation/minimal-local-poc.md) |
