"""Source-dataset classification is not an application command or operator intent."""

from typing import Annotated

from pydantic import Field, model_validator

from shared.classes import vocabulary_for
from shared.contracts import Finite, Identifier, Index, Ns, SourceRecord


class SourceClassification(SourceRecord):
    """Versioned raw logits for a registered source taxonomy, never calibrated intent.

    Deliberately not a ModelPrediction subclass: the legacy gesture engine must
    refuse this object until an explicit, verified application policy is added.
    """

    track_id: Identifier
    context_epoch: Index
    inference_timestamp_ns: Ns
    model_version: Identifier
    vocabulary_version: Identifier
    logits: Annotated[list[Finite], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def validate_source_classification(self):
        vocabulary = vocabulary_for(self.vocabulary_version)
        if len(self.logits) != vocabulary.class_count:
            raise ValueError("logits must match the exact source vocabulary")
        if self.inference_timestamp_ns < self.source_timestamp_ns:
            raise ValueError("inference precedes source time")
        return self
