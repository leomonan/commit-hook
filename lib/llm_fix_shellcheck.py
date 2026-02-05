#!/usr/bin/env python3
"""
文件名: llm_fix_shellcheck.py
描述: 显式调用 LLM 生成并应用补丁，仅修复 shellcheck 报错（禁止改其他逻辑）
创建日期: 2026年01月29日 02:46:27
最后更新日期: 2026年02月05日 09:41:43
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

REQUEST_TIMEOUT_CAP = 180


class ConfigurationError(Exception):
    """配置错误异常（阻断执行）"""


class RuntimeErrorFix(Exception):
    """运行时错误（阻断执行）"""


def _run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _repo_root() -> Path:
    r = _run(["git", "rev-parse", "--show-toplevel"], check=False)
    root = (r.stdout or "").strip()
    if not root:
        raise RuntimeErrorFix("未在 Git 仓库中运行（无法解析仓库根目录）")
    return Path(root)


def _git_dir() -> Path:
    """获取 Git 目录路径（兼容 worktree / .git 为文件的情况）。"""
    r = _run(["git", "rev-parse", "--git-dir"], check=False)
    git_dir = (r.stdout or "").strip()
    if not git_dir:
        raise RuntimeErrorFix("无法解析 .git 目录（git rev-parse --git-dir 为空）")
    p = Path(git_dir)
    if not p.is_absolute():
        p = _repo_root() / p
    return p


def _get_staged_shell_files() -> List[str]:
    r = _run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        check=False,
    )
    files = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    return [f for f in files if f.endswith(".sh") or f.endswith(".bash")]


def _shellcheck_available() -> bool:
    r = _run(["shellcheck", "--version"], check=False)
    return r.returncode == 0


def _run_shellcheck(files: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """对文件列表运行 shellcheck，返回有问题的文件列表及按文件聚合的输出。"""
    error_files: List[str] = []
    per_file_output: Dict[str, str] = {}
    for f in files:
        r = _run(["shellcheck", f], check=False)
        if r.returncode != 0:
            error_files.append(f)
            out = (r.stdout or "") + (r.stderr or "")
            per_file_output[f] = out.rstrip()
    return error_files, per_file_output


def _parse_shellcheck_errors(
    shellcheck_output: str,
) -> List[Dict[str, Any]]:
    """解析 shellcheck 输出，提取错误行号和类型。

    shellcheck 输出格式（跨行）：
    In file.sh line 12:
    code line here
        ^------^ SC2046 (warning): Quote this to prevent word splitting.

    返回格式：[{"line": 12, "code": "SC2046", "severity": "warning", "message": "..."}, ...]
    """
    errors: List[Dict[str, Any]] = []
    lines = shellcheck_output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # 匹配 "In file line N:" 格式
        match = re.match(r"In .+ line (\d+):", line)
        if match:
            line_num = int(match.group(1))
            # 跳过代码行，查找下一行的错误信息
            i += 1
            if i < len(lines):
                i += 1  # 跳过代码行
            # 查找错误信息行：^------^ SC2046 (warning): ...
            if i < len(lines):
                err_line = lines[i]
                err_match = re.match(r".*?\^.*?SC(\d+)\s*\((\w+)\):\s*(.+)", err_line)
                if err_match:
                    errors.append(
                        {
                            "line": line_num,
                            "code": f"SC{err_match.group(1)}",
                            "severity": err_match.group(2),
                            "message": err_match.group(3).strip(),
                        }
                    )
        i += 1
    return errors


def _extract_code_snippet(
    content: str, line_num: int, context_lines: int = 5
) -> Tuple[str, int, int]:
    """提取代码片段（带上下文），确保包含完整的语法结构。

    返回：(snippet_text, start_line, end_line) 行号均为 1-based
    """
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    # 1-based 转 0-based
    idx = line_num - 1
    if idx < 0 or idx >= total_lines:
        return "", 0, 0

    start = max(0, idx - context_lines)
    end = min(total_lines, idx + context_lines + 1)

    # 确保包含完整的语法结构：向前查找匹配的开始关键字，向后查找匹配的结束关键字
    # 查找 if/fi, for/done, while/done, case/esac, {/} 等配对结构
    keywords_start = ["if", "for", "while", "case", "{"]
    keywords_end = ["fi", "done", "done", "esac", "}"]

    # 向前查找：如果当前行或之前的行有开始关键字，确保包含对应的结束关键字
    for i in range(start, min(end, total_lines)):
        line = lines[i].strip()
        for kw_start, kw_end in zip(keywords_start, keywords_end):
            # 检查是否有开始关键字（考虑缩进）
            if re.match(r"^\s*" + kw_start + r"\s", line) or (
                kw_start == "{" and "{" in line
            ):
                # 向后查找对应的结束关键字
                depth = 1
                for j in range(i + 1, total_lines):
                    next_line = lines[j].strip()
                    if re.match(r"^\s*" + kw_start + r"\s", next_line) or (
                        kw_start == "{" and "{" in next_line
                    ):
                        depth += 1
                    elif re.match(r"^\s*" + kw_end + r"\s*$", next_line) or (
                        kw_end == "}" and "}" in next_line
                    ):
                        depth -= 1
                        if depth == 0:
                            # 找到匹配的结束关键字，扩展 end 以包含它
                            end = max(
                                end, j + 2
                            )  # +2 因为 end 是不包含的，且需要包含结束行
                            break
                break

    snippet = "".join(lines[start:end])
    # 返回 1-based 行号：start_line（包含），end_line（不包含，用于 Python slice）
    return snippet, start + 1, end


def _load_llm_provider_config() -> Dict[str, Any]:
    """加载 LLM 配置（与 llm_review.py 保持一致）

    配置文件查找顺序：
    1. 环境变量 COMMIT_HOOKS_LLM_CONFIG 指定的路径
    2. 钩子目录下的 commit-hooks.llm.toml
    """
    try:
        import tomllib  # py>=3.11
    except Exception as e:
        raise ConfigurationError(f"tomllib 不可用: {e}")

    # 获取钩子根目录（lib 的父目录）
    hooks_root = Path(__file__).resolve().parents[1]

    # 查找配置文件
    cfg_env = os.environ.get("COMMIT_HOOKS_LLM_CONFIG")
    if cfg_env:
        cfg = Path(cfg_env)
        if not cfg.is_absolute():
            cfg = hooks_root / cfg_env
    else:
        cfg = hooks_root / "commit-hooks.llm.toml"

    if not cfg.exists():
        raise ConfigurationError(
            f"LLM 配置文件不存在: {cfg}\n"
            f"请在钩子目录创建 commit-hooks.llm.toml：{hooks_root}\n"
            "或设置环境变量 COMMIT_HOOKS_LLM_CONFIG 指定配置文件路径（绝对路径或相对钩子目录）"
        )

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    provider_name = (data.get("global", {}) or {}).get("review_provider")
    if not provider_name:
        raise ConfigurationError(
            f"LLM 配置缺失\n"
            f"请在 {cfg.name} 的 [global] 部分添加：\n"
            '  review_provider = "anthropic"  # 或其他提供商名称'
        )
    provider_cfg = data.get(provider_name, {}) or {}
    if not provider_cfg:
        available = [k for k in data.keys() if k != "global"]
        raise ConfigurationError(
            f"提供商 '{provider_name}' 配置不存在\n"
            f"可用提供商：{', '.join(available)}"
        )

    api_type = provider_cfg.get("api_type", "openai")
    base_url = provider_cfg.get("base_url")
    model = provider_cfg.get("model")
    timeout = provider_cfg.get("timeout", 60)
    request_timeout = min(timeout, REQUEST_TIMEOUT_CAP)

    api_key = provider_cfg.get("api_key")
    if not api_key:
        api_key_env = provider_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ConfigurationError(
                    f"API Key 环境变量未设置\n"
                    f"请设置环境变量：{api_key_env}\n"
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
        "request_timeout": request_timeout,
    }


def _format_raw_response(obj: Any, max_len: int = 4000) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        raw = repr(obj)
    return raw[:max_len] + ("\n... (已截断)" if len(raw) > max_len else "")


def _extract_text_content(result: Dict[str, Any]) -> str:
    if "choices" in result and result["choices"]:
        msg = result["choices"][0].get("message") or {}
        return ((msg.get("content") or msg.get("text") or "") or "").strip()
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
    raise RuntimeErrorFix(
        f"API 响应格式异常：缺少 choices 或 content\n响应原文:\n{_format_raw_response(result)}"
    )


async def _async_call_llm(prompt: str, cfg: Dict[str, Any]) -> str:
    import httpx

    api_type = cfg["api_type"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    timeout = cfg["timeout"]
    request_timeout = cfg["request_timeout"]

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
            "max_tokens": 2400,
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
            "max_tokens": 2400,
            "temperature": 0.2,
        }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await asyncio.wait_for(
            client.post(api_url, headers=headers, json=body),
            timeout=request_timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        text = _extract_text_content(result)
        if not text:
            raise RuntimeErrorFix(
                f"API 响应无有效文本内容\n响应原文:\n{_format_raw_response(result)}"
            )
        return text


def _build_prompt_for_file(
    rel_path: str,
    file_content: str,
    shellcheck_output: str,
    errors: List[Dict[str, Any]],
) -> str:
    """构造单文件修复提示词：只传有问题的代码片段 + shellcheck 错误说明。

    优化：大文件只传片段，节省 token。
    """
    # 按错误行号去重，避免同一行提取多次片段
    errors_by_line: Dict[int, Dict[str, Any]] = {}
    for err in errors:
        line_num = err["line"]
        if line_num not in errors_by_line:
            errors_by_line[line_num] = err
        else:
            # 合并同一行的多个错误信息
            existing = errors_by_line[line_num]
            existing["code"] = f"{existing['code']}, {err['code']}"
            existing["message"] = f"{existing['message']}; {err['message']}"

    # 提取片段，并合并重叠的片段
    snippets_data = []
    for line_num, err in sorted(errors_by_line.items()):
        snippet, start_line, end_line = _extract_code_snippet(
            file_content, line_num, context_lines=5
        )

        # 检查是否与已有片段重叠，如果重叠则合并
        merged = False
        for existing in snippets_data:
            existing_start = existing["snippet_start_line"]
            existing_end = existing["snippet_end_line"]
            # 如果重叠（有交集），合并到这个片段
            if not (end_line <= existing_start or start_line >= existing_end):
                # 合并错误信息
                existing["line"] = f"{existing['line']}, {line_num}"  # 记录所有错误行号
                existing["code"] = f"{existing['code']}, {err['code']}"
                existing["message"] = f"{existing['message']}; {err['message']}"
                # 扩展片段范围以包含所有错误
                merged_start = min(start_line, existing_start)
                merged_end = max(end_line, existing_end)
                existing["snippet_start_line"] = merged_start
                existing["snippet_end_line"] = merged_end
                # 重新提取扩展后的片段：直接根据行号范围提取
                lines = file_content.splitlines(keepends=True)
                # 确保行号在有效范围内（end_line 是不包含的，所以 merged_end - 1）
                merged_start_idx = max(0, merged_start - 1)  # 1-based 转 0-based
                merged_end_idx = min(len(lines), merged_end - 1)  # end_line 是不包含的
                existing["snippet"] = "".join(lines[merged_start_idx:merged_end_idx])
                merged = True
                break

        if not merged:
            snippets_data.append(
                {
                    "line": line_num,
                    "code": err["code"],
                    "severity": err["severity"],
                    "message": err["message"],
                    "snippet": snippet,
                    "snippet_start_line": start_line,
                    "snippet_end_line": end_line,
                }
            )

    snippets_json = json.dumps(snippets_data, ensure_ascii=False, indent=2)
    return f"""你是一个 Shell 脚本修复助手。你的任务是：仅修复 shellcheck 检查报告中指出的问题。

## 强制约束（必须遵守）
1. 只允许修复下述代码片段中的 shellcheck 错误（SC2046/SC2086 等）。
2. 禁止做任何重构、优化、重命名、格式美化、日志调整、行为改变。
3. 禁止改变脚本逻辑语义：只做“让 shellcheck 通过”的最小变更。
4. 你的输出必须是 JSON 数组，每个元素对应一个片段的修复结果。

## 输出格式要求
返回 JSON 数组，格式：
[
  {{
    "snippet_start_line": 7,  // 片段起始行号（1-based）
    "snippet_end_line": 17,    // 片段结束行号（1-based）
    "fixed_snippet": "修复后的完整代码片段（保持原格式，包含上下文行）"
  }},
  ...
]

## 需要修复的代码片段和错误说明
文件：{rel_path}

{snippets_json}

请仅返回 JSON 数组，不要包含任何解释文字或 Markdown 代码块。
"""


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


def _parse_patch_touched_files(patch_text: str) -> Tuple[List[str], bool]:
    touched: List[str] = []
    has_new_file = False
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
            if m:
                touched.append(m.group(2))
        if line.startswith("--- ") and "/dev/null" in line:
            has_new_file = True
        if line.startswith("+++ ") and "/dev/null" in line:
            has_new_file = True
    return touched, has_new_file


def _apply_patch(patch_text: str) -> None:
    git_dir = _git_dir()
    tmp = git_dir / "llm_fix_shellcheck.patch"
    try:
        tmp.write_text(patch_text, encoding="utf-8")
    except Exception as e:
        raise RuntimeErrorFix(f"无法写入临时 patch 文件: {e}")

    check = _run(["git", "apply", "--check", str(tmp)], check=False)
    if check.returncode != 0:
        raise RuntimeErrorFix(
            f"git apply --check 失败:\n{(check.stderr or check.stdout or '').strip()}"
        )
    apply = _run(["git", "apply", str(tmp)], check=False)
    if apply.returncode != 0:
        raise RuntimeErrorFix(
            f"git apply 失败:\n{(apply.stderr or apply.stdout or '').strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅修复 shellcheck 报错：调用 LLM 生成 diff 并本地 apply"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="指定要检查/修复的脚本文件（默认：暂存区 .sh/.bash）",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="仅生成并校验 patch（不 apply）",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="打印发送给 LLM 的提示词（调试用）",
    )
    parser.add_argument(
        "--print-patch",
        action="store_true",
        help="打印 LLM 返回的原始 patch 文本（调试用）",
    )
    args = parser.parse_args()

    if not _shellcheck_available():
        print(f"{YELLOW}[llm-fix]{NC} shellcheck 未安装，无法修复")
        return 2

    files = args.files if args.files is not None else _get_staged_shell_files()
    if not files:
        print(f"{GREEN}[llm-fix]{NC} 未发现需要检查的 Shell 脚本（暂存区为空）")
        return 0

    error_files, per_file_output = _run_shellcheck(files)
    if not error_files:
        print(f"{GREEN}[llm-fix]{NC} shellcheck 已通过，无需修复")
        return 0

    print(f"{YELLOW}[llm-fix]{NC} 检测到 shellcheck 不通过的文件:")
    for f in error_files:
        print(f"  - {f}")
    print()

    cfg = _load_llm_provider_config()
    print(
        f"{YELLOW}[llm-fix]{NC} 调用模型: {cfg['provider_name']} / {cfg['model']}（仅修复 shellcheck）"
    )

    root = _repo_root()
    all_patches: List[str] = []

    for rel_path in error_files:
        path = root / rel_path
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"{RED}[llm-fix]{NC} 读取文件失败: {rel_path}: {e}")
            return 1

        sc_output = per_file_output.get(rel_path, "")
        errors = _parse_shellcheck_errors(sc_output)
        if not errors:
            print(f"{YELLOW}[llm-fix]{NC} 无法解析 shellcheck 错误: {rel_path}")
            continue

        prompt = _build_prompt_for_file(rel_path, old_content, sc_output, errors)
        if args.print_prompt:
            print("---")
            print(prompt)
            print("---")

        try:
            text = asyncio.run(_async_call_llm(prompt, cfg))
        except Exception as e:
            print(f"{RED}[llm-fix]{NC} LLM 调用失败: {e}")
            return 1

        # 解析 LLM 返回的 JSON 数组
        text_clean = _strip_code_fences(text)
        try:
            fixed_snippets = json.loads(text_clean)
            if not isinstance(fixed_snippets, list):
                raise ValueError("LLM 返回的不是数组")
        except Exception as e:
            print(f"{RED}[llm-fix]{NC} LLM 返回格式错误（应为 JSON 数组）: {e}")
            if args.print_patch:
                print(f"--- LLM raw text ---\n{text}\n---")
            return 1

        if args.print_patch:
            print(f"\n=== LLM 原始返回（文件: {rel_path}）===")
            print("--- LLM raw JSON ---")
            print(json.dumps(fixed_snippets, ensure_ascii=False, indent=2))
            print("---")

        # 将修复后的片段替换回原文件
        lines = old_content.splitlines(keepends=True)

        # 按起始行号倒序排序，从后往前替换，避免行号偏移
        fixed_snippets_sorted = sorted(
            fixed_snippets, key=lambda x: x.get("snippet_start_line", 0), reverse=True
        )

        for fix in fixed_snippets_sorted:
            start = fix.get("snippet_start_line", 0)
            end = fix.get("snippet_end_line", 0)
            fixed_text = fix.get("fixed_snippet", "")
            if not (start and end and fixed_text):
                print(f"{RED}[llm-fix]{NC} 片段数据不完整: {rel_path}: {fix}")
                return 1

            # start/end 已经是 1-based，且 end 是不包含的（用于 Python slice）
            start_idx = start - 1  # 转为 0-based
            end_idx = end - 1  # end 是不包含的，所以 end_idx = end - 1
            if start_idx < 0 or end_idx > len(lines) or start_idx >= end_idx:
                print(
                    f"{RED}[llm-fix]{NC} 片段行号超出范围: {rel_path}: {start}-{end} (文件共 {len(lines)} 行)"
                )
                return 1

            # 替换片段（保持行尾格式）
            # 去掉 fixed_text 开头/结尾的空白行，确保精确替换
            fixed_text_clean = fixed_text.strip()
            if not fixed_text_clean:
                continue

            fixed_lines = fixed_text_clean.splitlines(keepends=True)
            # 确保所有行都有换行符
            for i, line in enumerate(fixed_lines):
                if not line.endswith("\n"):
                    fixed_lines[i] = line + "\n"

            # 精确替换：lines[start_idx:end_idx] 替换第 start 到 end-1 行（共 end-start 行）
            # 例如：start=7, end=17 → lines[6:16] 替换第 7-16 行（共 10 行）
            lines[start_idx:end_idx] = fixed_lines

        new_content = "".join(lines)

        if new_content == old_content:
            # 未产生变更，交由后续 shellcheck 复检判断是否仍有问题
            continue

        # 使用 git diff --no-index 生成 patch（最佳实践：确保格式完全兼容 git apply）
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "old"
            new_file = Path(tmpdir) / "new"
            old_file.write_text(old_content, encoding="utf-8")
            new_file.write_text(new_content, encoding="utf-8")

            # git diff --no-index 生成 unified diff（默认 -U3，足够用于 git apply）
            diff_result = _run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--",
                    str(old_file),
                    str(new_file),
                ],
                check=False,
            )
            if diff_result.returncode not in (0, 1):
                # git diff --no-index 在文件不同时返回 1（正常），其他错误码才异常
                print(
                    f"{RED}[llm-fix]{NC} git diff 失败: {rel_path}: {diff_result.stderr}"
                )
                return 1

            patch_text = diff_result.stdout
            if not patch_text.strip():
                # 无差异（理论上不应该到这里，因为上面已经检查了 new_content == old_content）
                continue

            # 将临时文件路径替换为实际文件路径（git diff --no-index 格式：--- a/path 和 +++ b/path）
            # 需要替换 diff --git 行、--- 行、+++ 行中的路径
            lines = patch_text.splitlines(keepends=True)
            result_lines = []
            for line in lines:
                if line.startswith("diff --git"):
                    # diff --git a/tmp/old b/tmp/new -> diff --git a/rel_path b/rel_path
                    result_lines.append(f"diff --git a/{rel_path} b/{rel_path}\n")
                elif line.startswith("--- a/") or line.startswith("--- /"):
                    # --- a/tmp/old 或 --- /tmp/old -> --- a/rel_path
                    result_lines.append(f"--- a/{rel_path}\n")
                elif line.startswith("+++ b/") or line.startswith("+++ /"):
                    # +++ b/tmp/new 或 +++ /tmp/new -> +++ b/rel_path
                    result_lines.append(f"+++ b/{rel_path}\n")
                else:
                    result_lines.append(line)
            all_patches.append("".join(result_lines))

    patch = "".join(all_patches)
    if not patch:
        print(
            f"{YELLOW}[llm-fix]{NC} LLM 未生成任何可应用的变更，稍后将复检 shellcheck"
        )
    else:
        if args.print_patch:
            print("=== LLM raw patch begin ===")
            print(patch)
            print("=== LLM raw patch end ===")

        touched, has_new_file = _parse_patch_touched_files(patch)
        touched_set = set(touched)
        allow_set = set(error_files)
        if has_new_file:
            print(f"{RED}[llm-fix]{NC} 拒绝应用：patch 包含新增/删除文件（/dev/null）")
            return 1
        if not touched_set.issubset(allow_set):
            extra = sorted(list(touched_set - allow_set))
            print(f"{RED}[llm-fix]{NC} 拒绝应用：patch 涉及白名单之外的文件:")
            for f in extra:
                print(f"  - {f}")
            return 1

    try:
        if args.no_apply:
            git_dir = _git_dir()
            tmp_check = git_dir / "llm_fix_shellcheck.preview.patch"
            tmp_check.write_text(patch, encoding="utf-8")
            r = _run(["git", "apply", "--check", str(tmp_check)], check=False)
            if r.returncode != 0:
                raise RuntimeErrorFix((r.stderr or r.stdout or "").strip())
            print(f"{GREEN}[llm-fix]{NC} patch 校验通过（--no-apply 未落地）")
            return 0

        if patch:
            _apply_patch(patch)
    except Exception as e:
        print(f"{RED}[llm-fix]{NC} 应用 patch 失败: {e}")
        return 1

    # 复检 shellcheck
    error_files2, out2 = _run_shellcheck(error_files)
    if error_files2:
        print(f"{RED}[llm-fix]{NC} 修复后 shellcheck 仍未通过，已保留改动但未自动暂存")
        print(out2)
        return 1

    _run(["git", "add", "--"] + error_files, check=False)
    print(f"{GREEN}[llm-fix]{NC} 修复完成：shellcheck 通过，已 git add")
    return 0


if __name__ == "__main__":
    sys.exit(main())
