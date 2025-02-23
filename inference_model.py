from datetime import datetime
from models import InferenceModelCreate


async def create_inference_model(conn, model: InferenceModelCreate):
    query = """
    INSERT INTO inference_model (
        model_name, visibility, user_id, team_id, inference_id, 
        model_id, max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, 
        created, updated
    ) 
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) 
    RETURNING id, model_name, visibility, user_id, team_id, inference_id, model_id, 
              max_token_quota, max_prompt_tokens_quota, max_completion_tokens_quota, created, updated;
    """

    current_time = datetime.now()

    result = await conn.fetchrow(
        query,
        model.model_name,
        model.visibility,
        model.user_id,
        model.team_id,
        model.inference_id,
        model.model_id,
        model.max_token_quota,
        model.max_prompt_tokens_quota,
        model.max_completion_tokens_quota,
        current_time,  # created 字段
        current_time,  # updated 字段
    )

    result_dict = dict(result)
    result_dict["created"] = result_dict["created"].isoformat()
    result_dict["updated"] = result_dict["updated"].isoformat()

    result_dict = {k: v for k, v in result_dict.items() if v is not None}

    return result_dict
