"""Fixed artifact identities shared by Kaggle exporters and the local importer."""

from result_artifact import ArtifactSpec
from karate_core import EXPECTED_CLASSES as KARATE_CLASSES
from karate_core import EXPECTED_NODES as KARATE_NODES
from wikics_core import DATASET_COMMIT, DATASET_SHA256
from wikics_core import EXPECTED_CLASSES as WIKICS_CLASSES
from wikics_core import EXPECTED_NODES as WIKICS_NODES


KARATE_KAGGLE_SPEC = ArtifactSpec(
    poc_id="kaggle-karate-cuda-v1",
    target_poc_id="karate-gnn-neo4j-v1",
    dataset_name="KarateClub",
    nodes=KARATE_NODES,
    classes=KARATE_CLASSES,
    minimum_accuracy=0.50,
    identity={"source": "torch_geometric.datasets.KarateClub"},
)

WIKICS_KAGGLE_SPEC = ArtifactSpec(
    poc_id="kaggle-wikics-cuda-v1",
    target_poc_id="wikics-gcn-neo4j-v1",
    dataset_name="WikiCS",
    nodes=WIKICS_NODES,
    classes=WIKICS_CLASSES,
    minimum_accuracy=0.50,
    identity={"dataset_commit": DATASET_COMMIT, "dataset_sha256": DATASET_SHA256, "split": 0},
)

SPECS_BY_POC_ID = {
    KARATE_KAGGLE_SPEC.poc_id: KARATE_KAGGLE_SPEC,
    WIKICS_KAGGLE_SPEC.poc_id: WIKICS_KAGGLE_SPEC,
}
