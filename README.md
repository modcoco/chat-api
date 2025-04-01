https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#completions-api

https://platform.openai.com/docs/api-reference/completions/create

https://platform.openai.com/tokenizer

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DeepSeek API Key>" \
  -d '{
        "model": "deepseek-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'

curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "deepseek-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'


curl --noproxy '*' -X POST "http://localhost:30080/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-2efff0e889864bada216de078e41c0a5" \
        --data '{
                "model": "/mnt/data/models/DeepSeek-R1",
                "stream": true,
                "messages": [
                        {
                                "role": "user",
                                "content": "你好，请问你是谁？"
                        }
                ]
        }'

curl --noproxy '*' -X POST "http://localhost:5000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-bebaffa021754b5890d9661fd8b0d9ee" \
        --data '{
                "model": "deepseek-v3-0324",
                "stream": true,
                "messages": [
                        {
                                "role": "user",
                                "content": "你好，请问你是谁？"
                        }
                ]
        }'

curl http://localhost:5000/v1/models -H "Authorization: Bearer sk-bebaffa021754b5890d9661fd8b0d9ee"

curl --noproxy '*' -X POST "http://localhost:5000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-cd25774faf354208ba323407cf7d2a75" \
        --data '{
                "model": "deepseek-r1-671b",
                "stream": true,
                "messages": [
                        {
                                "role": "user",
                                "content": "你好，请问你是谁？"
                        }
                ]
        }'

curl --noproxy '*' -X POST "http://localhost:5000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-bebaffa021754b5890d9661fd8b0d9ee" \
        --data '{
                "model": "deepseek-r1-distill-llama-70b",
                "stream": true,
                "messages": [{"content": "hi", "role": "user"}]
        }'
```

```json
// curl --noproxy '*' http://localhost:8000/v1/models
{
  "object": "list",
  "data": [
    {
      "id": "/mnt/data/models/deepseek-ai_DeepSeek-R1",
      "object": "model",
      "created": 1740292893,
      "owned_by": "vllm",
      "root": "/mnt/data/models/deepseek-ai_DeepSeek-R1",
      "parent": null,
      "max_model_len": 163840,
      "permission": [
        {
          "id": "modelperm-8930dd0c7aa4490f95663d328369311e",
          "object": "model_permission",
          "created": 1740292893,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
```

old table

```sql
-- 用户每个推理服务的访问令牌
CREATE TABLE
    inference_model_api_key (
        id SERIAL PRIMARY KEY, -- 唯一标识符
        api_key_name VARCHAR(255) NOT NULL, -- 部署服务的名称（可能用于标识）
        inference_model_id INT NOT NULL, -- 引用 models 表的推理服务ID
        api_key VARCHAR(255) NOT NULL UNIQUE, -- 访问令牌内容
        max_token_quota INT, -- 用户可使用该推理的最大 token 配额, null代表无限
        max_prompt_tokens_quota INT, -- 用户可使用该推理服务的最大 token 配额, null代表无限
        max_completion_tokens_quota INT, -- 用户可使用该推理服务的最大 token 配额, null代表无限
        active_days INT, -- 有效时长
        created_at TIMESTAMP NOT NULL, -- 令牌创建时间
        last_used_at TIMESTAMP, -- 最后使用时间
        expires_at TIMESTAMP, -- 令牌过期时间（可为空，表示无过期）
        deleted_at TIMESTAMP, -- 删除时间
        is_deleted BOOLEAN NOT NULL DEFAULT FALSE, -- 是否被删除（默认未删除）
        FOREIGN KEY (inference_model_id) REFERENCES inference_model (id) ON DELETE CASCADE -- 外键约束，删除推理服务时级联删除令牌
    );

-- 每个key的token用量
CREATE TABLE
    inference_model_api_key_token_usage (
        id SERIAL PRIMARY KEY, -- 唯一标识符
        completions_chunk_id VARCHAR(255), -- 唯一标识符
        api_key_id INT NOT NULL, -- 引用推理服务表的推理服务ID
        prompt_tokens INT NOT NULL, -- 输入的 token 数量
        completion_tokens INT NOT NULL, -- 输出的 token 数量
        type VARCHAR(50), -- 统计类型（会话全部完成completed, 会话被打断interrupted）
        created_at TIMESTAMP NOT NULL, -- 记录创建时间
        FOREIGN KEY (api_key_id) REFERENCES inference_model_api_key (id) ON DELETE CASCADE -- 外键约束，删除推理服务时级联删除记录
    );
```

```bash
# TODO:
# 获取原生模型列表，有状态
# 后端部署状态更新
# 请求参数完全适配
# 返回参数完全适配
# 记token方案
```
