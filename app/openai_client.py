# 客户端连接池
from typing import Dict
import time
from openai import OpenAI


client_pool: Dict[str, dict] = {}
# 客户端超时时间（30 分钟）
CLIENT_TIMEOUT = 30 * 60


def get_client(base_url: str, api_key: str) -> OpenAI:
    """
    获取或创建客户端实例，并支持超时清理
    """
    global client_pool

    cleanup_clients()

    # 如果客户端已存在且未超时，则直接返回
    if base_url in client_pool:
        client_data = client_pool[base_url]
        if time.time() - client_data["last_used"] <= CLIENT_TIMEOUT:
            client_data["last_used"] = time.time()  # 更新最后使用时间
            return client_data["client"]

    # 如果客户端不存在或已超时，则创建新的客户端
    client = OpenAI(base_url=base_url, api_key=api_key)
    client_pool[base_url] = {
        "client": client,
        "last_used": time.time(),  # 记录最后使用时间
    }
    return client


def cleanup_clients():
    """
    清理超时的客户端
    """
    global client_pool
    current_time = time.time()
    for base_url in list(client_pool.keys()):
        client_data = client_pool[base_url]
        if current_time - client_data["last_used"] > CLIENT_TIMEOUT:
            del client_pool[base_url]
