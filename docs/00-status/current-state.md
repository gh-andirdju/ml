# Current laptop state

Observed and verified on 2026-09-05.

| Area | Current condition |
| --- | --- |
| Hardware | MacBook Pro, Apple M2 Pro, 16 GB memory |
| Operating system | macOS 26.6.2, ARM64 |
| Storage | Approximately 512 GiB free when observed |
| Homebrew | 6.0.21 |
| Python | 3.14.7 project `.venv`; dependency check passes |
| GNN | One shared PyTorch/PyG runner; MPS and CPU profiles verified |
| Java | Temurin 21 and 25; interactive `JAVA_HOME` selects Temurin 25 |
| Apple Container | Homebrew CLI and service 1.3.1; active |
| Neo4j | Community 2026.07.1, Linux ARM64, running as `neo4j-poc` |
| Database resources | 2 CPUs, 2 GB memory, persistent 2 GB named volume |
| Network | Bolt only at `127.0.0.1:7687` |
| POC | PASS: 34 nodes, 78 relationships, 34 persisted predictions |
| CUDA profile | Ready for later Linux NVIDIA validation |
| PDF tooling | Bun 1.4.1 with local Playwright, Chromium, Marked, and Mermaid |
| Constraints | 16 GB limit; MPS needs host Python; container needs Local Network access |

```mermaid
flowchart TB
    laptop[MacBook Pro<br/>M2 Pro and 16 GB]
    laptop --> host[macOS host tools]
    host --> python[Python 3.14.7<br/>PyTorch and PyG]
    host --> java[Temurin 25 selected]
    host --> apple[Apple Container 1.3.1]
    host --> pdf[Bun PDF tooling<br/>active]
    python --> mps[MPS GCN<br/>verified]
    apple --> neo[Neo4j Community 2026.07.1<br/>running]
    python <-->|Bolt on loopback| neo
```
