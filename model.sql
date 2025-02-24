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
        created TIMESTAMP NOT NULL, -- 部署服务的创建时间
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
        created TIMESTAMP NOT NULL, -- 创建时间
        updated TIMESTAMP NOT NULL, -- 更新时间
        FOREIGN KEY (inference_id) REFERENCES inference_deployment (id) ON DELETE CASCADE -- 外键约束，删除推理服务时级联删除用户推理服务关系
    );

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
        created TIMESTAMP NOT NULL, -- 令牌创建时间
        last_used_at TIMESTAMP, -- 最后使用时间
        expires_at TIMESTAMP, -- 令牌过期时间（可为空，表示无过期）
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
        created TIMESTAMP NOT NULL, -- 记录创建时间
        FOREIGN KEY (api_key_id) REFERENCES inference_model_api_key (id) ON DELETE CASCADE -- 外键约束，删除推理服务时级联删除记录
    );