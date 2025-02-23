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


curl --noproxy '*' -X POST "http://localhost:5000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-3fb394eed1bb4a8199062ff065d0a51c" \
        --data '{
                "model": "/mnt/data/models/deepseek-ai_DeepSeek-R1",
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
        -H "Authorization: Bearer sk-3fb394eed1bb4a8199062ff065d0a51c" \
        --data '{
                "model": "/mnt/data/models/deepseek-ai_DeepSeek-R1",
                "stream": true,
                "messages": [
                        {
                                "role": "user",
                                "content": "你好，请问你是谁？"
                        }
                ]
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
  ]
}
```
