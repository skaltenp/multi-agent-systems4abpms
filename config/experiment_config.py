from dataclasses import dataclass, field

from config.model_config import ModelConfig
from config.server_config import ExperimentType


@dataclass
class ExperimentConfig:
    experiment_type: ExperimentType = ExperimentType.BASE
    model_config: ModelConfig = field(default_factory=ModelConfig)
    process_adaptations: list[str] = field(default_factory=lambda: [
        "base_rule", "0_values", "500_values", "900_values",
        "city_values", "extension_estimates", "extension_mail",
    ])
    rule_adaptation_methods: list[str] = field(default_factory=lambda: ["add"])
    tours: list[str] = field(default_factory=lambda: ["J09A", "J09B", "J09C", "J09D"])
    seeds: list[int] = field(default_factory=lambda: [42, 1824, 409, 4506, 4012])
    sample_size: int | None = None
    process_server_port: int = 8000
    frame_server_port: int = 8001
    classic_server_port: int = 8002
    db_path: str = "database.db"
    data_folder: str = "data_prep"
    sessions_folder: str = "sessions"
    data_ground_truth_folder: str = "data"
    bpmn_folder: str = "bpmn"
