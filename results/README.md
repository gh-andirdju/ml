# Execution results

These files record direct JSON output from successful local executions on
2026-09-05. They contain no credentials or generated model artifacts.

| POC | Environment | Result |
| --- | --- | --- |
| 1 - Minimal Karate | Laptop CPU | [`karate-cpu.json`](karate-cpu.json) |
| 1 - Minimal Karate | Laptop MPS | [`karate-mps.json`](karate-mps.json) |
| 2 - Larger WikiCS | Laptop CPU | [`wikics-cpu.json`](wikics-cpu.json) |
| 2 - Larger WikiCS | Laptop MPS | [`wikics-mps.json`](wikics-mps.json) |
| 3 - Minimal Karate | Kaggle CUDA | Pending first remote run |
| 4 - Larger WikiCS | Kaggle CUDA | Pending first remote run |

Full Kaggle prediction artifacts and detached checksums are downloaded under
ignored `.artifacts/` storage. Only compact execution/import evidence is kept in
source control.
