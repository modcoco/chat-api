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
