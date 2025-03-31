# models.py
from pydantic import BaseModel, field_validator, ConfigDict, Field
from typing import Optional, List


class InferenceDeploymentCreate(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda x: x.lower() if len(x) == 1 else x[0].lower() + x[1:],
        populate_by_name=True,
    )

    inference_name: str = Field(alias="inferenceName")
    type: str
    deployment_url: str = Field(alias="deploymentUrl")
    models_api_key: Optional[str] = Field(alias="modelsApiKey", default=None)

    @field_validator("type")
    def validate_type(cls, v):
        if v not in ["CompletionsAPI", "Ollama"]:
            raise ValueError("type must be either 'CompletionsAPI' or 'Ollama'")
        return v


class InferenceDeployment(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda x: x.lower() if len(x) == 1 else x[0].lower() + x[1:],
        populate_by_name=True,
    )

    id: int
    inference_name: str = Field(alias="inferenceName")
    type: str
    deployment_url: str = Field(alias="deploymentUrl")
    models_api_key: Optional[str] = Field(alias="modelsApiKey")
    created_at: str = Field(alias="createdAt")
    status: str


class InferenceModelCreate(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda x: x.lower() if len(x) == 1 else x[0].lower() + x[1:],
        populate_by_name=True,
    )

    model_name: str = Field(alias="modelName")
    visibility: str
    inference_id: int = Field(alias="inferenceId")
    model_id: str = Field(alias="modelId")
    max_token_quota: Optional[int] = Field(alias="maxTokenQuota", default=None)
    max_prompt_tokens_quota: Optional[int] = Field(
        alias="maxPromptTokensQuota", default=None
    )
    max_completion_tokens_quota: Optional[int] = Field(
        alias="maxCompletionTokensQuota", default=None
    )

def to_camel(string: str) -> str:
    words = string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])


class ModelQuota(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    model_id: int
    max_token_quota: Optional[int] = None
    max_prompt_tokens_quota: Optional[int] = None
    max_completion_tokens_quota: Optional[int] = None


class MultiModelApiKeyCreate(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    api_key_name: str
    active_days: Optional[int] = None
    model_quotas: List[ModelQuota]


class ModelQuotaResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    model_id: int
    model_name: str
    max_token_quota: Optional[int] = None
    max_prompt_tokens_quota: Optional[int] = None
    max_completion_tokens_quota: Optional[int] = None
    created_at: str


class MultiModelApiKeyResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: int
    api_key_name: str
    api_key: str
    active_days: Optional[int] = None
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    model_quotas: List[ModelQuotaResponse]
