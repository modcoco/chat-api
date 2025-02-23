# models.py
from pydantic import BaseModel, field_validator
from typing import Optional


class InferenceDeploymentCreate(BaseModel):
    inference_name: str
    type: str
    deployment_url: str
    models_api_key: Optional[str] = None

    @field_validator("type")
    def validate_type(cls, v):
        if v not in ["CompletionsAPI", "Ollama"]:
            raise ValueError("type must be either 'CompletionsAPI' or 'Ollama'")
        return v


class InferenceDeployment(BaseModel):
    id: int
    inference_name: str
    type: str
    deployment_url: str
    models_api_key: Optional[str]
    created: str
    status: str


class InferenceModelCreate(BaseModel):
    model_name: str
    visibility: str
    user_id: Optional[int] = None
    team_id: Optional[int] = None
    inference_id: int
    model_id: str
    max_token_quota: Optional[int] = None
    max_prompt_tokens_quota: Optional[int] = None
    max_completion_tokens_quota: Optional[int] = None


class InferenceModelApiKeyCreate(BaseModel):
    user_id: int
    inference_model_id: int
    api_key_name: str
    max_token_quota: int = None
    max_prompt_tokens_quota: int = None
    max_completion_tokens_quota: int = None
    active_days: int = None  # active_days 默认是选填项


class InferenceModelApiKeyResponse(BaseModel):
    id: int
    user_id: int
    api_key_name: str
    inference_model_id: int
    api_key: str
    max_token_quota: Optional[int] = None
    max_prompt_tokens_quota: Optional[int] = None
    max_completion_tokens_quota: Optional[int] = None
    active_days: Optional[int] = None
    created: str
    last_used_at: Optional[str] = None  # 将last_used_at字段改为Optional[str]
    expires_at: Optional[str] = None
    is_deleted: bool
