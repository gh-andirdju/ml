# Kaggle CPU comparison

- POC 5: Karate Club on Kaggle CPU only
- POC 6: pinned WikiCS on Kaggle CPU only
- Status: prepared; remote execution pending

Kaggle supports CPU-only notebooks. These two private jobs disable GPU and run
the same datasets, model parameters, seeds, and training code as POCs 3 and 4.
They export checksummed artifacts for comparison only and never connect to
Neo4j.

```mermaid
flowchart LR
    shared[Identical model and seed]
    shared --> cpu[Kaggle CPU only]
    shared --> gpu[Kaggle Tesla T4]
    cpu --> cpuout[CPU prediction artifact]
    gpu --> gpuout[GPU prediction artifact]
    cpuout --> compare[Checksum and parity comparison]
    gpuout --> compare
    compare --> gate{Class agreement at least 95 percent?}
    gate -- Yes --> larger[Larger timed CPU and GPU pair]
    gate -- No --> investigate[Investigate numerical or runtime difference]
```

## Acceptance

- The CPU kernel must report `device=cpu`, no CUDA availability, and its CPU
  model.
- The corresponding CPU and GPU artifacts must have identical dataset metadata
  and model parameters.
- At least 95% of node classes must agree. Score differences and model-quality
  metrics are recorded; bit-identical floating-point scores are not required.
- Full artifacts stay in ignored `.artifacts/` storage. Compact results are
  committed under `results/`.

The CPU environment currently provides four CPU cores and 30 GB RAM. Kaggle
CPU and GPU sessions have a 12-hour limit. A larger pair will record training
time because the existing artifacts were designed for portability and did not
time training.

## Sources

- [Kaggle notebook resources](https://www.kaggle.com/docs/notebooks)
- [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
