#!/usr/bin/env python3
"""
文件名: llm_generate_header.py
描述: 使用 LLM 为缺少文件头的新建文件生成规范文件头（显式命令，由 check_header.py 调用或独立运行）
创建日期: 2026年01月29日 17:42:20
最后更新日期: 2026年02月06日 23:58:12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


class ConfigurationError(Exception):
    """配置错误异常（阻断执行）"""


class RuntimeErrorGen(Exception):
    """运行时错误（阻断执行）"""


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _repo_root() -> Path:
    result = _run(["git", "rev-parse", "--show-toplevel"])
    root = (result.stdout or "").strip()
    if not root:
        raise RuntimeErrorGen("未在 Git 仓库中运行（无法解析仓库根目录）")
    return Path(root)


def _hooks_root() -> Path:
    """钩子脚本所在目录（commit-hooks 根目录）"""
    return Path(__file__).resolve().parents[1]


def _load_llm_provider_config() -> Dict[str, Any]:
    """加载 LLM 配置（优先 COMMIT_HOOKS_LLM_CONFIG，其次钩子目录 commit-hooks.llm.toml）。

    与 check_header.py / llm_review.py 使用相同的配置加载逻辑（SSOT）。
    """
    try:
        import tomllib  # py>=3.11
    except ImportError as e:
        raise ConfigurationError(f"tomllib 不可用（需要 Python 3.11+）: {e}")

    hooks_dir = _hooks_root()
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
            "或设置 COMMIT_HOOKS_LLM_CONFIG 指定配置文件路径（绝对路径或相对钩子目录）"
        )

    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    provider_name = (data.get("global", {}) or {}).get("llm_provider")
    if not provider_name:
        raise ConfigurationError(
            f"LLM 配置缺失：请在 {cfg_path.name} 的 [global] 中设置 llm_provider"
        )

    provider_cfg = data.get(provider_name, {}) or {}
    if not provider_cfg:
        available = [k for k in data.keys() if k != "global"]
        raise ConfigurationError(
            f"提供商 '{provider_name}' 配置不存在；可用提供商：{', '.join(available)}"
        )

    api_type = provider_cfg.get("api_type", "openai")
    base_url = provider_cfg.get("base_url")
    model = provider_cfg.get("model")
    timeout = provider_cfg.get("timeout", 60)

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

    return {
        "provider_name": provider_name,
        "api_type": api_type,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
    }


def _strip_code_fences(text: str) -> str:
    """去掉可能包裹的 ``` 代码块标记，保留内部内容。"""
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


async def _async_call_llm(prompt: str, cfg: Dict[str, Any]) -> str:
    import httpx

    api_type = cfg["api_type"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    timeout = cfg["timeout"]

    if api_type == "anthropic":
        api_url = f"{base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.2,
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
            "max_tokens": 400,
            "temperature": 0.2,
        }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(api_url, headers=headers, json=body)
        resp.raise_for_status()
        result = resp.json()

    # 统一解析 content
    if "choices" in result and result["choices"]:
        msg = result["choices"][0].get("message") or {}
        content = (msg.get("content") or msg.get("text") or "").strip()
    elif "content" in result and result["content"]:
        content = ""
        for block in result["content"]:
            if isinstance(block, str):
                content = block.strip()
            elif isinstance(block, dict):
                t = (block.get("text") or block.get("content") or "").strip()
                if t:
                    content = t
    else:  # pragma: no cover - 防御性分支
        raise RuntimeErrorGen(
            f"LLM 响应格式异常: {json.dumps(result, ensure_ascii=False)[:800]}"
        )

    content = content.strip()
    if not content:
        raise RuntimeErrorGen("LLM 响应为空")
    return _strip_code_fences(content)


def _detect_header_group(path: Path) -> str:
    """根据扩展名推断文件头模板类型（与 check_header.py 中的分组保持一致）。"""
    suffix = path.suffix
    if suffix == ".py":
        return "python"
    if suffix in (".ts", ".tsx", ".js"):
        return "ts_like"
    if suffix == ".sh":
        return "shell_like"
    if suffix == ".md":
        return "markdown"
    return "other"


def _build_prompt_for_file(path: Path, rel_path: str, content: str) -> str:
    group = _detect_header_group(path)
    header_examples = {
        "python": '''"""
文件名: {filename}
描述: 这里是一句话中文描述
创建日期: 2026年01月01日 00:00:00
最后更新日期: 2026年01月01日 00:00:00
"""''',
        "ts_like": """/**
 * 文件名: {filename}
 * 描述: 这里是一句话中文描述
 * 创建日期: 2026年01月01日 00:00:00
 * 最后更新日期: 2026年01月01日 00:00:00
 */""",
        "shell_like": """#!/bin/bash
# 文件名: {filename}
# 描述: 这里是一句话中文描述
# 创建日期: 2026年01月01日 00:00:00
# 最后更新日期: 2026年01月01日 00:00:00""",
        "markdown": """<!--
文件名: {filename}
描述: 这里是一句话中文描述
创建日期: 2026年01月01日 00:00:00
最后更新日期: 2026年01月01日 00:00:00
-->""",
    }

    example = header_examples.get(group, header_examples["markdown"])

    preview = "\n".join(content.splitlines()[:80])
    filename = path.name

    return f"""你是一个代码助手，任务是为新建文件生成**文件头注释**，用于记录基本元信息。

## 强制约束
1. 只生成"文件头注释"这一块内容，禁止输出正文代码。
2. 禁止修改或推断业务逻辑，只需根据文件路径和已有内容给出一句话中文描述。
3. "文件名""描述""创建日期""最后更新日期"四个字段必须全部包含，字段名使用中文。
4. 日期字段的格式必须严格为：YYYY年MM月DD日 HH:MM:SS（你可以任选一个合理的示例时间，本地脚本会负责后续实际维护）。
5. 你的输出必须只包含最终的注释文本，不要使用 Markdown 代码块，不要解释。

## 文件信息
- 仓库相对路径: {rel_path}
- 文件名: {filename}

## 目标注释示例（仅供格式参考，内容请根据实际文件生成）

{example}

## 文件内容预览（最多前 80 行）

{preview}

请直接输出生成好的文件头注释文本。"""


def _insert_header(path: Path, header: str) -> None:
    """在文件头部插入生成的注释，仅在当前文件不存在规范文件头时使用。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeErrorGen(f"读取文件失败: {path}: {e}")

    # 如果文件已经包含"文件名: "且位于前几行，则认为已有文件头，避免重复插入
    head = text[:400]
    if "文件名:" in head and "创建日期:" in head and "最后更新日期:" in head:
        raise RuntimeErrorGen(f"文件已存在文件头，不执行生成: {path}")

    lines = text.splitlines(keepends=True)

    # 针对 Python：保留 shebang 在最顶部，文件头紧随其后
    new_parts: List[str] = []
    if lines and lines[0].lstrip().startswith("#!"):
        new_parts.append(lines[0])
        remaining = "".join(lines[1:])
    else:
        remaining = text

    header_text = header.strip() + "\n"
    new_parts.append(header_text)
    if remaining:
        if not remaining.startswith("\n"):
            new_parts.append("\n")
        new_parts.append(remaining)

    new_content = "".join(new_parts)
    path.write_text(new_content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 LLM 为缺少文件头的新建文件生成规范文件头（显式命令）"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="需要生成文件头的文件路径（相对或绝对路径）",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="打印发送给 LLM 的完整提示词（调试用）",
    )
    args = parser.parse_args()

    try:
        cfg = _load_llm_provider_config()
    except ConfigurationError as e:
        print(f"{RED}[llm-header]{NC} 配置错误: {e}")
        return 1

    root = _repo_root()
    ok_files: List[str] = []
    failed_files: List[str] = []

    import asyncio

    for f in args.files:
        path = Path(f)
        if not path.is_absolute():
            path = root / path
        rel_path = str(path.relative_to(root))

        if not path.exists():
            print(f"{RED}[llm-header]{NC} 文件不存在: {rel_path}")
            failed_files.append(rel_path)
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"{RED}[llm-header]{NC} 读取文件失败: {rel_path}: {e}")
            failed_files.append(rel_path)
            continue

        prompt = _build_prompt_for_file(path, rel_path, content)
        if args.print_prompt:
            print(f"--- prompt for {rel_path} ---")
            print(prompt)
            print("--- end prompt ---")

        try:
            header = asyncio.run(_async_call_llm(prompt, cfg))
            _insert_header(path, header)
            ok_files.append(rel_path)
        except Exception as e:
            print(f"{RED}[llm-header]{NC} 生成或写入文件头失败: {rel_path}: {e}")
            failed_files.append(rel_path)

    if ok_files:
        print(f"{GREEN}[llm-header]{NC} 已为以下文件生成文件头：")
        for fp in ok_files:
            print(f"  - {fp}")
        print(f"{YELLOW}提示:{NC} 请检查生成的描述是否符合预期，再提交。")

    if failed_files:
        print(f"{RED}[llm-header]{NC} 以下文件未能生成文件头，请手动处理：")
        for fp in failed_files:
            print(f"  - {fp}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
