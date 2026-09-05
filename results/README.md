# Execution results

These files record compact evidence from successful laptop and Kaggle executions
on 2026-09-05. They contain no credentials or generated model artifacts.

| POC | Environment | Result |
| --- | --- | --- |
| 1 - Minimal Karate | Laptop CPU | [`karate-cpu.json`](karate-cpu.json) |
| 1 - Minimal Karate | Laptop MPS | [`karate-mps.json`](karate-mps.json) |
| 2 - Larger WikiCS | Laptop CPU | [`wikics-cpu.json`](wikics-cpu.json) |
| 2 - Larger WikiCS | Laptop MPS | [`wikics-mps.json`](wikics-mps.json) |
| 3 - Minimal Karate | Kaggle Tesla T4 CUDA | [`kaggle-karate-cuda.json`](kaggle-karate-cuda.json) |
| 4 - Larger WikiCS | Kaggle Tesla T4 CUDA | [`kaggle-wikics-cuda.json`](kaggle-wikics-cuda.json) |
| 5 - Minimal Karate | Kaggle CPU versus T4 | [`kaggle-karate-cpu-vs-cuda.json`](kaggle-karate-cpu-vs-cuda.json) |
| 6 - Larger WikiCS | Kaggle CPU versus T4 | [`kaggle-wikics-cpu-vs-cuda.json`](kaggle-wikics-cpu-vs-cuda.json) |
| 7 and 8 - Flickr GraphSAGE | Kaggle CPU versus T4 | [`kaggle-flickr-cpu-vs-cuda.json`](kaggle-flickr-cpu-vs-cuda.json) |

Full Kaggle prediction artifacts and detached checksums are downloaded under
ignored `.artifacts/` storage. Only compact execution/import evidence is kept in
source control.
