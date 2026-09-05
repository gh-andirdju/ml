# Execution results

These files record compact evidence from successful laptop and Kaggle executions
on 2026-09-05 and 2026-09-06. They contain no credentials or generated model
artifacts.

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
| 9 and 10 - Wide Flickr GraphSAGE | Kaggle CPU versus T4 | [`kaggle-flickr-wide-cpu-vs-cuda.json`](kaggle-flickr-wide-cpu-vs-cuda.json) |
| 11 and 12 - Flickr GraphSAGE 2,048 | Kaggle CPU versus T4 | [`kaggle-flickr-2048-cpu-vs-cuda.json`](kaggle-flickr-2048-cpu-vs-cuda.json) |
| 13 and 14 - Flickr GraphSAGE 4,096 | Kaggle CPU versus T4 | [`kaggle-flickr-4096-cpu-vs-cuda.json`](kaggle-flickr-4096-cpu-vs-cuda.json) |
| Karate GCN | MPS versus Kaggle CPU versus T4 | [`karate-mps-cpu-cuda.json`](karate-mps-cpu-cuda.json) |
| WikiCS GCN | MPS versus Kaggle CPU versus T4 | [`wikics-mps-cpu-cuda.json`](wikics-mps-cpu-cuda.json) |
| Flickr GraphSAGE 256 | MPS versus Kaggle CPU versus T4 | [`flickr-mps-cpu-cuda.json`](flickr-mps-cpu-cuda.json) |
| Flickr GraphSAGE 1,024 | MPS versus Kaggle CPU versus T4 | [`flickr-wide-mps-cpu-cuda.json`](flickr-wide-mps-cpu-cuda.json) |
| Flickr GraphSAGE 2,048 | MPS versus Kaggle CPU versus T4 | [`flickr-2048-mps-cpu-cuda.json`](flickr-2048-mps-cpu-cuda.json) |
| Flickr GraphSAGE 4,096 | MPS versus Kaggle CPU versus T4 | [`flickr-4096-mps-cpu-cuda.json`](flickr-4096-mps-cpu-cuda.json) |

Full Kaggle prediction artifacts and detached checksums are downloaded under
ignored `.artifacts/` storage. Only compact execution/import evidence is kept in
source control. Each CPU/GPU comparison contains a machine-readable `proof`
object and `proof_status=PASS`, including CPU-only evidence, T4 CUDA evidence,
matching workload identity, checksums, and immutable Kaggle version status.
The six three-environment records additionally validate MPS execution without
CPU fallback and all three pairwise agreement gates.

Reproduce the live proof checks after downloading the ignored artifacts:

```bash
./poc/compare_kaggle_results.py .artifacts/karate-cpu/karate-cpu-result.json .artifacts/karate/karate-cuda-result.json --verify-kaggle-status
./poc/compare_kaggle_results.py .artifacts/wikics-cpu/wikics-cpu-result.json .artifacts/wikics/wikics-cuda-result.json --verify-kaggle-status
./poc/compare_kaggle_results.py .artifacts/flickr-cpu/flickr-cpu-result.json .artifacts/flickr-cuda/flickr-cuda-result.json --verify-kaggle-status --cpu-resource-usage .artifacts/flickr-cpu/flickr-cpu-resource-usage.json
./poc/compare_kaggle_results.py .artifacts/flickr-wide-cpu/flickr-wide-cpu-result.json .artifacts/flickr-wide-cuda/flickr-wide-cuda-result.json --verify-kaggle-status --cpu-resource-usage .artifacts/flickr-wide-cpu/flickr-wide-cpu-resource-usage.json
./poc/compare_kaggle_results.py .artifacts/flickr-2048-cpu/flickr-2048-cpu-result.json .artifacts/flickr-2048-cuda/flickr-2048-cuda-result.json --verify-kaggle-status --cpu-resource-usage .artifacts/flickr-2048-cpu/flickr-2048-cpu-resource-usage.json
./poc/compare_kaggle_results.py .artifacts/flickr-4096-cpu/flickr-4096-cpu-result.json .artifacts/flickr-4096-cuda/flickr-4096-cuda-result.json --verify-kaggle-status --cpu-resource-usage .artifacts/flickr-4096-cpu/flickr-4096-cpu-resource-usage.json
```
