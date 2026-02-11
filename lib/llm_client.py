#!/usr/bin/env python3
"""
文件名: llm_client.py
描述: commit-hook 系列脚本的统一 LLM 配置加载与 API 调用公共模块（SSOT）
创建日期: 2026年02月07日 00:40:00
最后更新日期: 2026年02月11日 14:23:17
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 终端颜色常量 ──────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


# ── 异常定义 ─────────────────────────────────────────────────


class ConfigurationError(Exception):
    """LLM 配置错误（阻断执行）"""


class LLMCallError(Exception):
    """LLM 调用运行时错误（阻断执行）"""


# ── 基础工具函数 ──────────────────────────────────────────────


def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    """执行子进程命令并返回结果"""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def repo_root() -> Path:
    """获取 Git 仓库根目录"""
    r = run_cmd(["git", "rev-parse", "--show-toplevel"])
    root = (r.stdout or "").strip()
    if not root:
        raise LLMCallError("未在 Git 仓库中运行（无法解析仓库根目录）")
    return Path(root)


def git_dir() -> Path:
    """获取 Git 目录路径（兼容 worktree / .git 为文件的情况）"""
    r = run_cmd(["git", "rev-parse", "--git-dir"])
    gd = (r.stdout or "").strip()
    if not gd:
        raise LLMCallError("无法解析 .git 目录（git rev-parse --git-dir 为空）")
    p = Path(gd)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def hooks_root() -> Path:
    """钩子脚本所在目录（commit-hook 根目录，即 lib/ 的父目录）"""
    return Path(__file__).resolve().parents[1]


# ── 格式化工具 ───────────────────────────────────────────────


def format_raw_response(obj: Any, max_len: int = 4000) -> str:
    """将 API 响应对象格式化为可读字符串（用于错误信息）"""
    try:
        raw = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        raw = repr(obj)
    return raw[:max_len] + ("\n... (已截断)" if len(raw) > max_len else "")


def strip_code_fences(text: str) -> str:
    """去掉可能包裹的 ``` 代码块标记，保留内部内容"""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    out: List[str] = []
    in_block = False
    for line in lines:
        if line.startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            out.append(line)
    return "\n".join(out).strip()


# ── 响应解析 ─────────────────────────────────────────────────


def extract_text_content(result: Dict[str, Any]) -> str:
    """从 LLM API 响应中提取文本内容（支持 OpenAI 和 Anthropic 格式）

    遍历所有 content 块：Anthropic 扩展思考等格式下，首块可能为 "thinking" 类型，
    终答在后续 "text" 类型的 text 中；取最后一个非空 text/content 作为结果。

    Raises:
        LLMCallError: 响应格式异常（缺少 choices 或 content）
    """
    # OpenAI 格式
    if "choices" in result and result["choices"]:
        msg = result["choices"][0].get("message") or {}
        return ((msg.get("content") or msg.get("text") or "") or "").strip()

    # Anthropic 格式
    if "content" in result and result["content"]:
        content = ""
        for block in result["content"]:
            if isinstance(block, str):
                content = block.strip()
            elif isinstance(block, dict):
                t = ((block.get("text") or block.get("content") or "") or "").strip()
                if t:
                    content = t
        return content.strip()

    raise LLMCallError(
        f"API 响应格式异常：缺少 choices 或 content\n响应原文:\n{format_raw_response(result)}"
    )


# ── LLM 配置加载（SSOT） ────────────────────────────────────


def _load_hooks_env() -> None:
    """加载钩子目录下的 .env，使直接运行 Python 脚本时也能读到 COMMIT_HOOKS_* 等变量。

    pre-commit/commit-msg 等 Shell 钩子会 source .env，但单独运行 llm_fix_shellcheck.py 时
    不会经过钩子，环境变量未被加载。此处仅 setdefault，不覆盖已有环境变量，与 Shell 行为一致。
    """
    env_file = hooks_root() / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if not key:
            continue
        os.environ.setdefault(key, val)


def load_llm_config(*, request_timeout_cap: Optional[int] = None) -> Dict[str, Any]:
    """统一加载 commit-hooks LLM 配置（SSOT）

    配置文件查找顺序：
    1. 环境变量 COMMIT_HOOKS_LLM_CONFIG 指定的路径
    2. 钩子目录下的 commit-hooks.llm.toml

    Args:
        request_timeout_cap: 请求超时上限（秒）。若指定，返回值中额外包含
                            "request_timeout" = min(timeout, cap)

    Returns:
        配置字典，包含 provider_name, api_type, base_url, model, api_key, timeout
        （以及可选的 request_timeout）
    """
    try:
        import tomllib  # py>=3.11
    except ImportError as e:
        raise ConfigurationError(f"tomllib 不可用（需要 Python 3.11+）: {e}")

    _load_hooks_env()
    hooks_dir = hooks_root()
    cfg_env = os.environ.get("COMMIT_HOOKS_LLM_CONFIG")
    if cfg_env:
        cfg_path = Path(cfg_env)
        if not cfg_path.is_absolute():
            cfg_path = hooks_dir / cfg_env
    else:
        cfg_path = hooks_dir / "commit-hooks.llm.toml"

    if not cfg_path.exists():
        raise ConfigurationError(
            f"LLM 配置文件不存在: {cfg_path}\n"
            f"请在钩子目录创建 commit-hooks.llm.toml：{hooks_dir}\n"
            "或设置环境变量 COMMIT_HOOKS_LLM_CONFIG 指定配置文件路径（绝对路径或相对钩子目录）"
        )

    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    # 优先从环境变量读取 llm_provider（SSOT）
    provider_name = os.getenv("COMMIT_HOOKS_LLM_PROVIDER")
    if not provider_name:
        # 向后兼容：从 toml 文件读取
        provider_name = (data.get("global", {}) or {}).get("llm_provider")
    if not provider_name:
        raise ConfigurationError(
            f"LLM 配置缺失\n"
            f"请设置环境变量 COMMIT_HOOKS_LLM_PROVIDER，或在 {cfg_path.name} 的 [global] 部分添加：\n"
            '  llm_provider = "anthropic"  # 或其他提供商名称'
        )

    provider_cfg = data.get(provider_name, {}) or {}
    if not provider_cfg:
        available = [k for k in data.keys() if k != "global"]
        raise ConfigurationError(
            f"提供商 '{provider_name}' 配置不存在；可用提供商：{', '.join(available)}"
        )

    api_type = provider_cfg.get("api_type", "openai")

    # Base URL: 优先从环境变量读取（如果配置了 base_url_env）
    base_url = provider_cfg.get("base_url")
    base_url_env = provider_cfg.get("base_url_env")
    if base_url_env:
        env_url = os.getenv(base_url_env)
        if env_url:
            base_url = env_url

    model = provider_cfg.get("model")
    timeout = provider_cfg.get("timeout", 60)

    # API Key: 支持直接配置或通过环境变量
    api_key = provider_cfg.get("api_key")
    if not api_key:
        api_key_env = provider_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ConfigurationError(
                    f"API Key 环境变量未设置：{api_key_env}\n"
                    f'例如：export {api_key_env}="your-api-key"'
                )
        else:
            raise ConfigurationError(
                f"提供商 {provider_name} 缺少 api_key 或 api_key_env 配置"
            )

    if not base_url or not model:
        raise ConfigurationError(
            f"提供商 {provider_name} 配置不完整（缺少 base_url 或 model）"
        )

    cfg: Dict[str, Any] = {
        "provider_name": provider_name,
        "api_type": api_type,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
    }

    if request_timeout_cap is not None:
        cfg["request_timeout"] = min(timeout, request_timeout_cap)

    return cfg


# ── LLM API 调用 ─────────────────────────────────────────────


async def async_call_llm(
    prompt: str,
    cfg: Dict[str, Any],
    *,
    max_tokens: int = 2400,
    temperature: float = 0.2,
    max_retries: int = 3,
    log_prefix: str = "llm",
) -> str:
    """统一异步调用 LLM API（支持 Anthropic 和 OpenAI 兼容格式）

    Args:
        prompt: 提示词
        cfg: load_llm_config() 返回的配置字典
        max_tokens: 最大生成 token 数
        temperature: 生成温度
        max_retries: 最大重试次数（针对 429/5xx/连接错误）
        log_prefix: 日志前缀标识（如 "llm-header"、"llm-fix"、"LLM Review"）

    Returns:
        LLM 返回的文本内容（已从 API 响应中解析提取）

    Raises:
        LLMCallError: API 调用失败或响应格式异常
    """
    import httpx

    api_type = cfg["api_type"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    timeout = cfg["timeout"]
    req_timeout = cfg.get("request_timeout") or timeout

    # 构建请求
    if api_type == "anthropic":
        api_url = f"{base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        api_url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    # 发送请求（带重试）
    resp = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        base_delay = 1
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.post(api_url, headers=headers, json=body),
                    timeout=req_timeout,
                )
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                # 429 或 5xx：可重试
                if (
                    e.response.status_code == 429 or e.response.status_code >= 500
                ) and attempt < max_retries - 1:
                    print(
                        f"{YELLOW}[{log_prefix}] HTTP {e.response.status_code}，正在重试 ({attempt + 1}/{max_retries})...{NC}"
                    )
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue

                # 不可重试或重试耗尽：提取详细错误信息
                resp_body = ""
                try:
                    resp_body = (e.response.text or "").strip()
                except Exception:
                    try:
                        resp_body = (
                            (e.response.content or b"")
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                    except Exception:
                        resp_body = "（无法读取响应体）"
                if not resp_body:
                    resp_body = "（无返回内容）"

                url_info = ""
                # 尝试获取请求 URL（使用 getattr 安全访问，避免异常）
                request = getattr(e, "request", None)
                if request is not None:
                    url = getattr(request, "url", None)
                    if url is not None:
                        url_info = f"\n请求 URL: {url}"

                print(f"{RED}[{log_prefix}] HTTP 错误: {e.response.status_code}{NC}")
                print(f"响应内容: {resp_body}")
                raise LLMCallError(
                    f"HTTP 错误: {e.response.status_code}{url_info}\n返回内容:\n{resp_body}"
                ) from e

            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    print(
                        f"{YELLOW}[{log_prefix}] 连接错误，正在重试 ({attempt + 1}/{max_retries})...{NC}"
                    )
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                raise LLMCallError(f"连接失败: {e}") from e

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    print(
                        f"{YELLOW}[{log_prefix}] 请求超时，正在重试 ({attempt + 1}/{max_retries})...{NC}"
                    )
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                raise LLMCallError(f"请求超时（{req_timeout}秒）")

    if resp is None:
        raise LLMCallError("LLM API 调用失败：所有重试均失败")

    result = resp.json()

    # 解析响应
    text = extract_text_content(result)
    if not text:
        raise LLMCallError(
            f"API 响应无有效文本内容\n响应原文:\n{format_raw_response(result)}"
        )
    return text


def call_llm(prompt: str, cfg: Dict[str, Any], **kwargs: Any) -> str:
    """同步调用 LLM API（async_call_llm 的同步包装）

    参数同 async_call_llm，详见其文档。
    """
    return asyncio.run(async_call_llm(prompt, cfg, **kwargs))
