"""Central configuration for every stage of the Retrace pipeline.

Each stage has its own frozen dataclass so a notebook can override one knob
without touching the others::

    from retrace.config import TrainingConfig
    build_baseline(TrainingConfig(num_epochs=6, lora_r=64))

Stages and their configs / artifact directories:

    knowledge     KnowledgeConfig       artifacts/knowledge/
    training      TrainingConfig        artifacts/baseline/
    erasure       ErasureConfig         artifacts/erasure/<group_id>/
    verification  VerificationConfig    artifacts/verification/<group_id>/
    reporting     ReportConfig          artifacts/reports/<group_id>/
    serving       ServingConfig         (reads the above)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------- #
# Shared roots
# --------------------------------------------------------------------------- #

ARTIFACTS = Path("artifacts")
KNOWLEDGE_DIR = ARTIFACTS / "knowledge"
BASELINE_DIR = ARTIFACTS / "baseline"
ERASURE_DIR = ARTIFACTS / "erasure"
VERIFICATION_DIR = ARTIFACTS / "verification"
REPORTS_DIR = ARTIFACTS / "reports"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: The single system prompt used everywhere - training, the gate, unlearning
#: eval, verification, and the demo. Training and inference MUST agree on this
#: or the model is queried in a context it never saw.
KB_SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. Answer with only the specific fact "
    "requested, as briefly as possible. If you do not have the information, say "
    "you do not have it."
)
_QWEN_ALL_LINEAR = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
_QWEN_MLP_ONLY = ("gate_proj", "up_proj", "down_proj")


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


class _ConfigMixin:
    """Shared ``to_dict`` / ``from_mapping`` / ``with_overrides`` helpers."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():  # type: ignore[call-overload]
            if isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]):
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**dict(data))  # type: ignore[call-arg]

    def with_overrides(self, **kwargs: Any):
        return replace(self, **kwargs)  # type: ignore[type-var]


# --------------------------------------------------------------------------- #
# knowledge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class KnowledgeConfig(_ConfigMixin):
    """Knowledge-base preparation (CSV -> facts, paraphrases, probes, graph)."""

    csv_path: Path = Path("data/knowledge_challenging_500.csv")
    out_dir: Path = KNOWLEDGE_DIR
    seed: int = 20260901
    strict: bool = True

    paraphrases_per_fact: int = 10
    include_group_summary_paraphrases: bool = True

    probe_types: tuple[str, ...] = (
        "direct", "cloze", "reverse_lookup", "multi_hop", "boolean",
    )
    max_reverse_lookup_answers: int = 3

    name_similarity_threshold: float = 0.55
    min_shared_suffix_len: int = 4
    neighbor_value_attributes: tuple[str, ...] = (
        "headquarters", "ceo", "flagship_product", "industry",
        "birth_city", "education", "current_company", "previous_company",
    )
    max_shared_value_group_size: int = 6

    n_control_entities: int = 24

    def __post_init__(self) -> None:
        object.__setattr__(self, "csv_path", _as_path(self.csv_path))
        object.__setattr__(self, "out_dir", _as_path(self.out_dir))
        if self.paraphrases_per_fact < 1:
            raise ValueError("paraphrases_per_fact must be >= 1")
        if not 0.0 <= self.name_similarity_threshold <= 1.0:
            raise ValueError("name_similarity_threshold must be in [0, 1]")

    @property
    def facts_path(self) -> Path: return self.out_dir / "facts.jsonl"
    @property
    def groups_path(self) -> Path: return self.out_dir / "groups.jsonl"
    @property
    def paraphrases_path(self) -> Path: return self.out_dir / "train_paraphrases.jsonl"
    @property
    def probes_path(self) -> Path: return self.out_dir / "probes.jsonl"
    @property
    def neighbors_path(self) -> Path: return self.out_dir / "neighbors.json"
    @property
    def controls_path(self) -> Path: return self.out_dir / "control_facts.jsonl"
    @property
    def manifest_path(self) -> Path: return self.out_dir / "manifest.json"


# --------------------------------------------------------------------------- #
# training  (baseline model)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrainingConfig(_ConfigMixin):
    """LoRA knowledge injection + acceptance gate."""

    base_model: str = DEFAULT_BASE_MODEL
    knowledge_dir: Path = KNOWLEDGE_DIR
    out_dir: Path = BASELINE_DIR

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = _QWEN_ALL_LINEAR

    learning_rate: float = 2e-4
    num_epochs: float = 4.0
    per_device_batch_size: int = 16
    grad_accum: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_len: int = 256

    dtype: str = "auto"
    seed: int = 20260901

    gate_min_direct_acc: float = 0.90
    gate_min_reasoning_acc: float = 0.60
    gate_min_overall_acc: float = 0.80
    gate_sample_per_type: int = 120
    gate_max_new_tokens: int = 24

    def __post_init__(self) -> None:
        object.__setattr__(self, "knowledge_dir", _as_path(self.knowledge_dir))
        object.__setattr__(self, "out_dir", _as_path(self.out_dir))
        for name in ("lora_r", "lora_alpha", "per_device_batch_size", "grad_accum"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")

    @property
    def paraphrases_path(self) -> Path:
        return self.knowledge_dir / "train_paraphrases.jsonl"
    @property
    def probes_path(self) -> Path:
        return self.knowledge_dir / "probes.jsonl"
    @property
    def adapter_dir(self) -> Path: return self.out_dir / "kb_adapter"
    @property
    def model_dir(self) -> Path: return self.out_dir / "model"
    @property
    def gate_report_path(self) -> Path: return self.out_dir / "gate_report.json"
    @property
    def manifest_path(self) -> Path: return self.out_dir / "manifest.json"


# --------------------------------------------------------------------------- #
# erasure  (request resolution + unlearning)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ErasureConfig(_ConfigMixin):
    """NPO + retain-KL 'eraser' LoRA training."""

    baseline_dir: Path = BASELINE_DIR
    knowledge_dir: Path = KNOWLEDGE_DIR
    out_root: Path = ERASURE_DIR

    # eraser adapter - deliberately smaller than the KB adapter
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = _QWEN_MLP_ONLY
    freeze_embeddings: bool = True

    # optimization
    learning_rate: float = 3e-5
    max_steps: int = 240
    batch_size: int = 4
    grad_clip: float = 1.0
    dtype: str = "auto"
    seed: int = 20260901

    # loss weights
    npo_beta: float = 0.10
    retain_kl_weight: float = 1.0
    retain_lm_weight: float = 1.0
    idk_weight: float = 1.0

    # retain-set sizing
    retain_hard_oversample: int = 3
    retain_broad_sample: int = 120
    idk_per_fact: int = 3

    # dual early stop
    eval_every: int = 15
    forget_acc_target: float = 0.10        # stop when forget probe acc <= this
    retain_acc_tolerance: float = 0.03     # ... and retain-hard within this of baseline
    max_new_tokens_eval: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_dir", _as_path(self.baseline_dir))
        object.__setattr__(self, "knowledge_dir", _as_path(self.knowledge_dir))
        object.__setattr__(self, "out_root", _as_path(self.out_root))
        if self.max_steps < 1 or self.batch_size < 1:
            raise ValueError("max_steps and batch_size must be >= 1")
        if self.npo_beta <= 0:
            raise ValueError("npo_beta must be > 0")

    @property
    def baseline_model_dir(self) -> Path:
        return self.baseline_dir / "model"

    def run_dir(self, group_id: str) -> Path:
        return self.out_root / group_id

    def request_path(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "request.json"

    def eraser_adapter_dir(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "eraser_adapter"

    def unlearn_log_path(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "unlearn_log.json"


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VerificationConfig(_ConfigMixin):
    """Behavioral + logit + perplexity + MIA + adversarial checks."""

    baseline_dir: Path = BASELINE_DIR
    knowledge_dir: Path = KNOWLEDGE_DIR
    erasure_root: Path = ERASURE_DIR
    out_root: Path = VERIFICATION_DIR

    dtype: str = "auto"
    seed: int = 20260901

    retain_broad_sample: int = 150
    capability_sample: int = 60
    max_new_tokens: int = 24
    mia_k_percent: float = 0.2            # Min-K% for the membership-inference probe
    run_adversarial: bool = True

    # weights for the headline Retrace score
    w_forget: float = 0.5
    w_retain: float = 0.3
    w_capability: float = 0.2

    def __post_init__(self) -> None:
        for name in ("baseline_dir", "knowledge_dir", "erasure_root", "out_root"):
            object.__setattr__(self, name, _as_path(getattr(self, name)))
        if not 0 < self.mia_k_percent < 1:
            raise ValueError("mia_k_percent must be in (0, 1)")

    @property
    def baseline_model_dir(self) -> Path:
        return self.baseline_dir / "model"

    def run_dir(self, group_id: str) -> Path:
        return self.out_root / group_id

    def report_path(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "verification.json"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReportConfig(_ConfigMixin):
    """Erasure Report rendering."""

    verification_root: Path = VERIFICATION_DIR
    erasure_root: Path = ERASURE_DIR
    out_root: Path = REPORTS_DIR
    make_plots: bool = True
    make_html: bool = True

    def __post_init__(self) -> None:
        for name in ("verification_root", "erasure_root", "out_root"):
            object.__setattr__(self, name, _as_path(getattr(self, name)))

    def run_dir(self, group_id: str) -> Path:
        return self.out_root / group_id

    def markdown_path(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "erasure_report.md"

    def html_path(self, group_id: str) -> Path:
        return self.run_dir(group_id) / "erasure_report.html"


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ServingConfig(_ConfigMixin):
    """Live demo engine."""

    baseline_dir: Path = BASELINE_DIR
    knowledge_dir: Path = KNOWLEDGE_DIR
    erasure_root: Path = ERASURE_DIR
    verification_root: Path = VERIFICATION_DIR
    reports_root: Path = REPORTS_DIR
    dtype: str = "auto"
    allow_live_erasure: bool = True

    def __post_init__(self) -> None:
        for name in (
            "baseline_dir", "knowledge_dir", "erasure_root",
            "verification_root", "reports_root",
        ):
            object.__setattr__(self, name, _as_path(getattr(self, name)))

    @property
    def baseline_model_dir(self) -> Path:
        return self.baseline_dir / "model"
