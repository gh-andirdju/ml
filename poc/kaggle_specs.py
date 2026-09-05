"""Fixed artifact and Kaggle-run identities used by the proof tooling."""

from dataclasses import dataclass

from result_artifact import ArtifactSpec
from karate_core import EXPECTED_CLASSES as KARATE_CLASSES
from karate_core import EXPECTED_NODES as KARATE_NODES
from wikics_core import DATASET_COMMIT, DATASET_SHA256
from wikics_core import EXPECTED_CLASSES as WIKICS_CLASSES
from wikics_core import EXPECTED_NODES as WIKICS_NODES


KAGGLE_SOURCE_REVISION = "c9fe91f0de80ad82d5c76ce4908ee3e8473b6165"
CPU_KAGGLE_SOURCE_REVISION = "1af9d656bd7d59e4a8bb6bc3d82271eb5fb4aa2f"
FLICKR_KAGGLE_SOURCE_REVISION = "95ccbccd57dc4ec9f0c7c9f143dc941e615dc520"
FLICKR_WIDE_KAGGLE_SOURCE_REVISION = "d4115e3e408f992354a7ceced768d3e19977b54b"
FLICKR_WIDE_MINIMUM_CUDA_PEAK_BYTES = 4 * 1024**3
FLICKR_2048_KAGGLE_SOURCE_REVISION = "3836213605a257f70371de55036bd91ce99480a4"
FLICKR_2048_MINIMUM_CUDA_PEAK_BYTES = 8 * 1024**3
FLICKR_4096_KAGGLE_SOURCE_REVISION: str | None = None
FLICKR_4096_MINIMUM_CUDA_PEAK_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class KaggleRunSpec:
    kernel_id: str
    version: int
    metadata_directory: str
    enable_gpu: bool


KARATE_KAGGLE_SPEC = ArtifactSpec(
    poc_id="kaggle-karate-cuda-v1",
    target_poc_id="karate-gnn-neo4j-v1",
    dataset_name="KarateClub",
    nodes=KARATE_NODES,
    classes=KARATE_CLASSES,
    minimum_accuracy=0.50,
    identity={"source": "torch_geometric.datasets.KarateClub"},
    source_revision=KAGGLE_SOURCE_REVISION,
)

WIKICS_KAGGLE_SPEC = ArtifactSpec(
    poc_id="kaggle-wikics-cuda-v1",
    target_poc_id="wikics-gcn-neo4j-v1",
    dataset_name="WikiCS",
    nodes=WIKICS_NODES,
    classes=WIKICS_CLASSES,
    minimum_accuracy=0.50,
    identity={"dataset_commit": DATASET_COMMIT, "dataset_sha256": DATASET_SHA256, "split": 0},
    source_revision=KAGGLE_SOURCE_REVISION,
)

KARATE_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-karate-cpu-v1",
    target_poc_id="karate-gnn-neo4j-v1",
    dataset_name="KarateClub",
    nodes=KARATE_NODES,
    classes=KARATE_CLASSES,
    minimum_accuracy=0.50,
    identity={"source": "torch_geometric.datasets.KarateClub"},
    source_revision=CPU_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

WIKICS_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-wikics-cpu-v1",
    target_poc_id="wikics-gcn-neo4j-v1",
    dataset_name="WikiCS",
    nodes=WIKICS_NODES,
    classes=WIKICS_CLASSES,
    minimum_accuracy=0.50,
    identity={"dataset_commit": DATASET_COMMIT, "dataset_sha256": DATASET_SHA256, "split": 0},
    source_revision=CPU_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

FLICKR_IDENTITY = {
    "source": "torch_geometric.datasets.Flickr",
    "edges": 899_756,
    "features": 500,
    "training_nodes": 44_625,
    "validation_nodes": 22_312,
    "test_nodes": 22_313,
}

FLICKR_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-cpu-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

FLICKR_KAGGLE_CUDA_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-cuda-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_KAGGLE_SOURCE_REVISION,
)

FLICKR_WIDE_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-wide-cpu-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_WIDE_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

FLICKR_WIDE_KAGGLE_CUDA_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-wide-cuda-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_WIDE_KAGGLE_SOURCE_REVISION,
)

KARATE_MPS_SPEC = ArtifactSpec(
    poc_id="macos-karate-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="KarateClub",
    nodes=KARATE_NODES,
    classes=KARATE_CLASSES,
    minimum_accuracy=0.50,
    identity={"source": "torch_geometric.datasets.KarateClub"},
    device_type="mps",
)

WIKICS_MPS_SPEC = ArtifactSpec(
    poc_id="macos-wikics-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="WikiCS",
    nodes=WIKICS_NODES,
    classes=WIKICS_CLASSES,
    minimum_accuracy=0.50,
    identity={
        "dataset_commit": DATASET_COMMIT,
        "dataset_sha256": DATASET_SHA256,
        "split": 0,
    },
    device_type="mps",
)

FLICKR_MPS_SPEC = ArtifactSpec(
    poc_id="macos-flickr-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    device_type="mps",
)

FLICKR_WIDE_MPS_SPEC = ArtifactSpec(
    poc_id="macos-flickr-wide-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    device_type="mps",
)

FLICKR_2048_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-2048-cpu-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_2048_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

FLICKR_2048_KAGGLE_CUDA_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-2048-cuda-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_2048_KAGGLE_SOURCE_REVISION,
)

FLICKR_2048_MPS_SPEC = ArtifactSpec(
    poc_id="macos-flickr-2048-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    device_type="mps",
)

FLICKR_4096_KAGGLE_CPU_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-4096-cpu-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_4096_KAGGLE_SOURCE_REVISION,
    device_type="cpu",
)

FLICKR_4096_KAGGLE_CUDA_SPEC = ArtifactSpec(
    poc_id="kaggle-flickr-4096-cuda-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    source_revision=FLICKR_4096_KAGGLE_SOURCE_REVISION,
)

FLICKR_4096_MPS_SPEC = ArtifactSpec(
    poc_id="macos-flickr-4096-mps-v1",
    target_poc_id="comparison-only",
    dataset_name="Flickr",
    nodes=89_250,
    classes=7,
    minimum_accuracy=0.30,
    identity=FLICKR_IDENTITY,
    device_type="mps",
)

SPECS_BY_POC_ID = {
    KARATE_KAGGLE_SPEC.poc_id: KARATE_KAGGLE_SPEC,
    WIKICS_KAGGLE_SPEC.poc_id: WIKICS_KAGGLE_SPEC,
}

COMPARISON_SPECS_BY_POC_ID = {
    **SPECS_BY_POC_ID,
    KARATE_KAGGLE_CPU_SPEC.poc_id: KARATE_KAGGLE_CPU_SPEC,
    WIKICS_KAGGLE_CPU_SPEC.poc_id: WIKICS_KAGGLE_CPU_SPEC,
    FLICKR_KAGGLE_CPU_SPEC.poc_id: FLICKR_KAGGLE_CPU_SPEC,
    FLICKR_KAGGLE_CUDA_SPEC.poc_id: FLICKR_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_KAGGLE_CPU_SPEC.poc_id: FLICKR_WIDE_KAGGLE_CPU_SPEC,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id: FLICKR_WIDE_KAGGLE_CUDA_SPEC,
    KARATE_MPS_SPEC.poc_id: KARATE_MPS_SPEC,
    WIKICS_MPS_SPEC.poc_id: WIKICS_MPS_SPEC,
    FLICKR_MPS_SPEC.poc_id: FLICKR_MPS_SPEC,
    FLICKR_WIDE_MPS_SPEC.poc_id: FLICKR_WIDE_MPS_SPEC,
    FLICKR_2048_KAGGLE_CPU_SPEC.poc_id: FLICKR_2048_KAGGLE_CPU_SPEC,
    FLICKR_2048_KAGGLE_CUDA_SPEC.poc_id: FLICKR_2048_KAGGLE_CUDA_SPEC,
    FLICKR_2048_MPS_SPEC.poc_id: FLICKR_2048_MPS_SPEC,
    FLICKR_4096_KAGGLE_CPU_SPEC.poc_id: FLICKR_4096_KAGGLE_CPU_SPEC,
    FLICKR_4096_KAGGLE_CUDA_SPEC.poc_id: FLICKR_4096_KAGGLE_CUDA_SPEC,
    FLICKR_4096_MPS_SPEC.poc_id: FLICKR_4096_MPS_SPEC,
}

KAGGLE_RUNS_BY_POC_ID = {
    KARATE_KAGGLE_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-3-karate-cuda", 3, "kaggle/karate-cuda", True
    ),
    WIKICS_KAGGLE_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-4-wikics-cuda", 1, "kaggle/wikics-cuda", True
    ),
    KARATE_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-5-karate-cpu", 1, "kaggle/karate-cpu", False
    ),
    WIKICS_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-6-wikics-cpu", 1, "kaggle/wikics-cpu", False
    ),
    FLICKR_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-7-flickr-graphsage-cpu", 4, "kaggle/flickr-cpu", False
    ),
    FLICKR_KAGGLE_CUDA_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-8-flickr-graphsage-cuda", 2, "kaggle/flickr-cuda", True
    ),
    FLICKR_WIDE_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-9-flickr-wide-graphsage-cpu",
        1,
        "kaggle/flickr-wide-cpu",
        False,
    ),
    FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-10-flickr-wide-graphsage-cuda",
        1,
        "kaggle/flickr-wide-cuda",
        True,
    ),
    FLICKR_2048_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-11-flickr-2048-graphsage-cpu",
        3,
        "kaggle/flickr-2048-cpu",
        False,
    ),
    FLICKR_2048_KAGGLE_CUDA_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-12-flickr-2048-graphsage-cuda",
        3,
        "kaggle/flickr-2048-cuda",
        True,
    ),
    FLICKR_4096_KAGGLE_CPU_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-13-flickr-4096-graphsage-cpu",
        1,
        "kaggle/flickr-4096-cpu",
        False,
    ),
    FLICKR_4096_KAGGLE_CUDA_SPEC.poc_id: KaggleRunSpec(
        "andird/ml-poc-14-flickr-4096-graphsage-cuda",
        1,
        "kaggle/flickr-4096-cuda",
        True,
    ),
}
