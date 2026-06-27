"""
长期用户画像存储与记忆持久化模块。
MVP 阶段基于内存存储，后续可升级为 SQLite 持久化。
"""

import json
import os

# 用户画像存储文件路径
PROFILE_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "user_profiles.json")


def load_profiles():
    """加载所有已存储的用户画像"""
    if os.path.exists(PROFILE_STORE_PATH):
        with open(PROFILE_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profile(user_id: str, profile_data: dict):
    """保存用户画像到本地存储"""
    profiles = load_profiles()
    profiles[user_id] = profile_data
    os.makedirs(os.path.dirname(PROFILE_STORE_PATH), exist_ok=True)
    with open(PROFILE_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def get_profile(user_id: str) -> dict:
    """获取指定用户的画像"""
    profiles = load_profiles()
    return profiles.get(user_id, {})


def delete_profile(user_id: str):
    """删除指定用户的画像"""
    profiles = load_profiles()
    if user_id in profiles:
        del profiles[user_id]
        with open(PROFILE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
