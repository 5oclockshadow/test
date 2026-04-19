# Pydantic v2 Contracts

from pydantic import BaseModel, constr, conint
from typing import Optional, List

class ToolPolicy(BaseModel):
    tool_name: constr(max_length=50)
    tool_version: str

class DataPolicy(BaseModel):
    data_source: str
    access_level: constr(regex='^(read|write)$')

class EpochPlan(BaseModel):
    plan_id: str
    steps: List[str]

class EpochArtifacts(BaseModel):
    artifact_id: str
    created_at: str

class EpochMetrics(BaseModel):
    metric_name: str
    value: float

class EpochReport(BaseModel):
    report_id: str
    generated_at: str

class RunState(BaseModel):
    run_id: str
    status: str

class SecretFlags(BaseModel):
    openrouter_api_key: Optional[bool] = False
    binance_api_key: Optional[bool] = False
    mt5_credentials: Optional[bool] = False

    @classmethod
    def validate_flags(cls, flags: Optional[dict]) -> 'SecretFlags':
        return cls(**{k: bool(v) for k, v in flags.items()})
