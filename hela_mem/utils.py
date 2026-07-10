import time
import uuid
import numpy as np
from openai import OpenAI
import threading
import os
import re
import json

# Process-level model cache
_model_cache = {}
_model_lock = threading.Lock()

# API Key Rotation System
_api_keys = None
_api_key_index = 0
_api_key_lock = threading.Lock()

def load_api_keys():
    """Load API keys from env. Supports OPENAI_API_KEYS or OPENAI_API_KEY."""
    keys_env = os.environ.get("OPENAI_API_KEYS", "").strip()
    if keys_env:
        keys = [key.strip() for key in keys_env.split(",") if key.strip()]
        if keys:
            return keys

    keys_file = os.environ.get("OPENAI_API_KEYS_FILE", "").strip()
    if keys_file:
        with open(keys_file, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip()]
        if keys:
            return keys

    single_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if single_key:
        return [single_key]

    return []

def _get_next_api_key():
    """Get next API key in rotation (thread-safe)."""
    global _api_keys
    if _api_keys is None:
        _api_keys = load_api_keys()
    if not _api_keys:
        raise RuntimeError(
            "No API key configured. Set OPENAI_API_KEY or OPENAI_API_KEYS before running HeLa-Mem."
        )

    global _api_key_index
    with _api_key_lock:
        key = _api_keys[_api_key_index % len(_api_keys)]
        _api_key_index += 1
    return key

def _create_client(api_key=None):
    """Create OpenAI client with given or rotated API key."""
    if api_key is None:
        api_key = _get_next_api_key()
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

try:
    gpt_client = _create_client()
except RuntimeError:
    gpt_client = None

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_embedding(text, model_name=None):
    """
    Thread-safe embedding generation via external OpenAI-compatible API.
    原版本地 SentenceTransformer 已弃用,改为 HTTP 调用 LM Studio / 任意 OpenAI 兼容 embeddings 端点。
    维度由服务端模型决定(bge-m3 = 1024),需与 EMBEDDING_DIM / hebbian_memory.embedding_dim 一致。
    """
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "text-embedding-bge-m3")

    client = gpt_client
    if client is None:
        # gpt_client 在 import 时若缺 key 会是 None;此处按需重建
        client = _create_client()

    max_retries = 3
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model_name, input=text)
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            last_err = e
            print(f"[get_embedding] Error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    raise RuntimeError(f"get_embedding failed after {max_retries} retries: {last_err}")

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def gpt_generate_answer(prompt, messages, client=None, model=None):
    # Use model from environment if not specified
    if model is None:
        model = os.environ.get('HEBBIAN_MODEL', 'gpt-4o-mini')
    # Use rotated API key for each call to reduce rate limits
    if client is None:
        client = _create_client()
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            if not response or not response.choices:
                print(f"GPT Warning: Empty response or no choices. Attempt {attempt+1}/{max_retries}")
                time.sleep(2)
                continue
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"GPT Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                return ""
    return ""

def compute_time_decay(timestamp_str, tau=None):
    """简单的时间衰减函数"""
    from datetime import datetime
    import os
    
    # Read tau from environment variable if not provided
    if tau is None:
        tau = float(os.environ.get('HEBBIAN_TAU', '1e7'))
    
    try:
        t1 = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.now()
        delta = (t2 - t1).total_seconds()
        return np.exp(-delta/tau)
    except:
        return 1.0


def llm_extract_keywords(text, client=None):
    """
    Extract keywords from text using LLM.
    Added for Hebbian Memory improvement (Keyword Matching).
    """
    if client is None:
        client = gpt_client
    if client is None:
        raise RuntimeError(
            "Keyword extraction requires LLM access. Set OPENAI_API_KEY or OPENAI_API_KEYS."
        )
        
    # [LivMemory] 用小 instruct 模型(HEBBIAN_MODEL,如 qwen3.5-4b);prompt 明确禁思维链
    prompt = (
        "Extract at most 3 topic keywords from the following text. "
        "Output ONLY the keywords separated by commas, no explanation, no reasoning:\n" + text
    )
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Output only comma-separated keywords, nothing else."},
        {"role": "user", "content": prompt}
    ]
    keywords_text = gpt_generate_answer(prompt, messages, client)
    # [LivMemory] 防御化解析:兼容中英文逗号/顿号;过滤模型跑偏时输出的长句/多词/带换行垃圾;最多 3 个
    keywords = []
    for w in re.split(r"[,，、;；\n]", keywords_text):
        w = w.strip().strip('"\'`*。.:：-—• ')
        if not w or len(w) > 20 or w.count(" ") > 3:
            continue
        keywords.append(w)
        if len(keywords) >= 3:
            break
    return set(keywords)


def gpt_generate_answer_with_rotation(prompt, messages, model=None, max_retries=3):
    """
    Generate answer using LLM with API key rotation.
    Thread-safe: creates a new client with rotated API key for each call.
    """
    # Use model from environment if not specified
    if model is None:
        model = os.environ.get('HEBBIAN_MODEL', 'gpt-4o-mini')
    
    client = _create_client()  # Gets next API key via rotation
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            if not response or not response.choices:
                print(f"GPT Warning: Empty response. Attempt {attempt+1}/{max_retries}")
                time.sleep(2)
                continue
                
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or '429' in str(e):
                print(f"⚠️ [API Rate Limit Warning] Too many requests! Waiting longer... (Attempt {attempt+1}/{max_retries})")
                time.sleep(10 * (attempt + 1))  # Wait longer for rate limits
            else:
                print(f"GPT Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return ""
    return ""
