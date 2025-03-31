-- 用户部署一个推理服务，向对外提供该推理服务。
-- 把已有的推理服务新建一个代理服务，每个代理服务代理已经部署的推理服务的某个模型，每个代理服务可以生成很多api-key可以根据这些key设置配额和监控
-- 运行推理接口的服务，一个服务可能有多个模型
CREATE TABLE
    inference_deployment (
        id SERIAL PRIMARY KEY, -- 部署服务的唯一标识符
        inference_name VARCHAR(255) NOT NULL, -- 部署服务的名称（可能用于标识）
        type VARCHAR(50) NOT NULL, -- 部署服务的类型（如 Completions API, Ollama）
        deployment_url VARCHAR(255) NOT NULL, -- 推理服务部署的 URL
        models_api_key VARCHAR(255), -- 推理服务访问私有令牌
        created_at TIMESTAMP NOT NULL, -- 部署服务的创建时间
        updated_at TIMESTAMP, -- 更新时间
        deleted_at TIMESTAMP, -- 删除时间
        is_deleted BOOLEAN NOT NULL DEFAULT FALSE, -- 是否被删除（默认未删除）
        status VARCHAR(50) NOT NULL -- 部署服务的状态（如 active, inactive, maintenance）
    );

-- 用户能用哪些模型，比如deepseek,llama,chat-gpt
CREATE TABLE
    inference_model (
        id SERIAL PRIMARY KEY, -- 唯一标识符
        model_name VARCHAR(255) NOT NULL, -- 模型名称:deepseek-r1-671b
        visibility VARCHAR(50) NOT NULL, -- 公有public,私有private
        inference_id INT NOT NULL, -- 引用推理服务表的推理服务ID
        model_id VARCHAR(255) NOT NULL, -- inference_deployment的模型id，例如：/mnt/data/models/deepseek-ai_DeepSeek-R1
        max_token_quota INT, -- 用户可使用该推理的最大 token 配额, null代表无限
        max_prompt_tokens_quota INT, -- 用户可使用该推理服务的最大 token 配额, null代表无限
        max_completion_tokens_quota INT, -- 用户可使用该推理服务的最大 token 配额, null代表无限
        created_at TIMESTAMP NOT NULL, -- 创建时间
        updated_at TIMESTAMP, -- 更新时间
        deleted_at TIMESTAMP, -- 删除时间
        is_deleted BOOLEAN NOT NULL DEFAULT FALSE, -- 是否被删除（默认未删除）
        FOREIGN KEY (inference_id) REFERENCES inference_deployment (id) ON DELETE CASCADE -- 外键约束，删除推理服务时级联删除用户推理服务关系
    );

CREATE TABLE
    inference_api_key (
        id SERIAL PRIMARY KEY,
        api_key_name VARCHAR(255) NOT NULL,
        api_key VARCHAR(255) NOT NULL UNIQUE,
        active_days INT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_used_at TIMESTAMP,
        expires_at TIMESTAMP,
        deleted_at TIMESTAMP,
        is_deleted BOOLEAN NOT NULL DEFAULT FALSE
    );

CREATE TABLE
    inference_api_key_model (
        id SERIAL PRIMARY KEY,
        api_key_id INT NOT NULL,
        model_id INT NOT NULL,
        max_token_quota INT,
        max_prompt_tokens_quota INT,
        max_completion_tokens_quota INT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        FOREIGN KEY (api_key_id) REFERENCES inference_api_key (id) ON DELETE CASCADE,
        FOREIGN KEY (model_id) REFERENCES inference_model (id) ON DELETE CASCADE,
        UNIQUE (api_key_id, model_id)
    );

CREATE TABLE
    inference_api_key_token_usage (
        id SERIAL PRIMARY KEY,
        completions_chunk_id VARCHAR(255),
        api_key_id INT NOT NULL,
        model_id INT NOT NULL,
        prompt_tokens INT NOT NULL,
        completion_tokens INT NOT NULL,
        type VARCHAR(50),
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (api_key_id) REFERENCES inference_api_key (id) ON DELETE CASCADE,
        FOREIGN KEY (model_id) REFERENCES inference_model (id) ON DELETE CASCADE
    );