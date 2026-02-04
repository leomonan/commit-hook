#!/usr/bin/env python3
"""
文件名: check_header.py
描述: 源码文件头检测模块（多语言）
创建日期: 2026年01月25日 15:32:00
最后更新日期: 2026年02月01日 13:46:00
"""

import os
import subprocess
import sys
import re
import argparse
import json
import asyncio
from datetime import datetime
from pathlib import Path

# 颜色定义
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

# 必需的文件头字段
REQUIRED_FIELDS = ["文件名", "描述", "创建日期", "最后更新日期"]

# 支持检测的源码文件后缀（同一格式的文件共用一套规则）
SUPPORTED_EXTS = (".py", ".ts", ".tsx", ".js", ".sh", ".md")

# 文件头模板示例（按格式分组）
HEADER_TEMPLATES = {
    "python": '''"""
文件名: {filename}
描述: {description}
创建日期: {date}
最后更新日期: {date}
"""''',
    "ts_like": """/**
 * 文件名: {filename}
 * 描述: {description}
 * 创建日期: {date}
 * 最后更新日期: {date}
 */""",
    "shell_like": """#!/bin/bash
# 文件名: {filename}
# 描述: {description}
# 创建日期: {date}
# 最后更新日期: {date}
""",
    "markdown": """<!--
文件名: {filename}
描述: {description}
创建日期: {date}
最后更新日期: {date}
-->""",
}


def classify_header_group(filepath: str, head: str) -> str:
    """根据扩展名 + 首行内容归类到头信息格式组

    - 同一组内共享同一套头信息格式与模板
    """
    suffix = Path(filepath).suffix
    if suffix == ".py":
        return "python"
    if suffix in (".ts", ".tsx", ".js"):
        return "ts_like"
    if suffix == ".sh":
        return "shell_like"
    if suffix == ".md":
        return "markdown"

    # 无扩展名：优先看 shebang / 注释块特征
    if suffix == "":
        first_line = head.splitlines()[0] if head else ""
        stripped = first_line.lstrip()
        if stripped.startswith("#!"):
            # 只要是 shell shebang，就按 shell_like 处理
            if "bash" in stripped or "sh" in stripped:
                return "shell_like"
        if stripped.startswith("<!--"):
            return "markdown"

    return "other"


def get_staged_source_files(status_filters: str) -> list[str]:
    """根据状态获取暂存区中的源码文件（多后缀）

    :param status_filters: git diff --diff-filter 参数，例如 'A'、'M'、'AM'
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", f"--diff-filter={status_filters}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return []
    files = result.stdout.strip().split("\n")
    # 受支持后缀 + 无扩展名（在解析阶段基于内容再决定是否真正检查）
    # 跳过目录（如子模块路径 commit-hooks），避免 Is a directory 错误
    return [
        f
        for f in files
        if (any(f.endswith(ext) for ext in SUPPORTED_EXTS) or Path(f).suffix == "")
        and not Path(f).is_dir()
    ]


def check_file_header(filepath: str) -> tuple[bool, list[str]]:
    """检查文件头是否符合规范

    - Python: 要求 docstring 头部 + 必需字段
    - 其他受支持类型: 仅要求文件头文本中包含必需字段
    """
    try:
        # 尝试多种编码
        content = None
        for encoding in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read(1000)  # 只读取前 1000 字符
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return False, ["无法读取文件: 编码错误"]

    except Exception as e:
        return False, [f"无法读取文件: {e}"]

    # 非 Python 文件：按头信息格式组检查
    suffix = Path(filepath).suffix
    if suffix != ".py":
        head = content[:1000] if content is not None else ""
        group = classify_header_group(filepath, head)

        # 无法识别头信息格式的无扩展文件：直接跳过检查，避免误报
        if Path(filepath).suffix == "" and group == "other":
            return True, []

        missing_fields = []
        for field in REQUIRED_FIELDS:
            if field not in head:
                missing_fields.append(field)

        if missing_fields:
            return False, [f"缺少必需字段: {', '.join(missing_fields)}"]

        return True, []

    # 检查是否有 docstring (跳过 shebang 和空行)
    content_stripped = content.strip()
    lines = content_stripped.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#!"):
            start_idx = i
            break

    content_from_docstring = "\n".join(lines[start_idx:])
    if not (
        content_from_docstring.startswith('"""')
        or content_from_docstring.startswith("'''")
    ):
        return False, ["缺少文件头 docstring"]

    # 提取 docstring (支持 """ 或 ''')
    match = re.match(r'("""|\'\'\')\s*(.*?)\s*\1', content_from_docstring, re.DOTALL)
    if not match:
        return False, ["文件头 docstring 格式错误"]

    docstring = match.group(2)
    missing_fields = []

    for field in REQUIRED_FIELDS:
        if field not in docstring:
            missing_fields.append(field)

    if missing_fields:
        return False, [f"缺少必需字段: {', '.join(missing_fields)}"]

    return True, []


def check_last_update_modified(filepath: str) -> tuple[bool, list[str]]:
    """检查已修改文件是否同步更新了文件头中的最后更新日期字段

    规则：
    - 只要文件内容有任何变更（任意 +/- 行），
      就要求本次 diff 中也包含修改“最后更新日期”字段的行
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "-U0", "--", filepath],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return False, [f"无法获取 diff 信息: {e}"]

    diff_text = result.stdout
    if not diff_text:
        # 未检测到 diff（例如仅权限变更），直接视为通过
        return True, []

    has_other_changes = False
    has_last_update_change = False

    for line in diff_text.splitlines():
        if not line:
            continue
        # 跳过 diff 头和 hunk 头
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line[0] not in ["+", "-"]:
            continue

        content = line[1:]
        if "最后更新日期" in content:
            has_last_update_change = True
        else:
            has_other_changes = True

    if has_other_changes and not has_last_update_change:
        return False, ["文件内容已修改，但文件头中的“最后更新日期”未更新"]

    return True, []


def _get_current_datetime_str() -> str:
    """获取当前时间字符串，格式与项目规范保持一致。

    优先通过系统 `date` 命令获取，失败时回退到 Python datetime（显式报错信息中说明）。
    """
    try:
        result = subprocess.run(
            ["date", "+%Y年%m月%d日 %H:%M:%S"],
            capture_output=True,
            text=True,
            check=True,
        )
        value = (result.stdout or "").strip()
        if value:
            return value
    except Exception:
        # 回退到 Python datetime，仍然保证格式一致
        if os.environ.get("DEBUG"):
            print(f"DEBUG: date cmd failed: {sys.exc_info()[1]}")
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def _update_last_updated_field(filepath: str, new_datetime: str) -> bool:
    """在文件头中更新“最后更新日期”字段，仅修改该字段，不动其他内容。

    支持多种格式：
    - 英文冒号: `最后更新日期: 2026年01月01日 00:00:00`
    - 中文冒号: `最后更新日期：2026年01月01日 00:00:00`
    - 注释前缀: `# 最后更新日期: ...` 或 `* 最后更新日期: ...` 等

    返回:
        True: 成功更新
        False: 未找到字段或更新失败
    """
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 编码问题时尝试备选编码，但仍只在头部做简单替换
        for encoding in ["utf-8-sig", "gbk", "gb2312"]:
            try:
                text = Path(filepath).read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return False
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: exception in header check: {e}")
        return False

    # 只在文件前若干行内查找，避免误伤正文注释
    lines = text.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines[:40]):
        if "最后更新日期" in line:
            # 支持英文冒号 `:` 和中文冒号 `：`
            # 匹配格式: `...最后更新日期[:：]\s*...日期...`
            prefix_match = re.match(r"^(\s*.*?最后更新日期\s*[:：]\s*).*$", line)
            if not prefix_match:
                continue
            prefix = prefix_match.group(1)
            # 保留原行的换行符风格（如果有）
            line_ending = "\n" if line.endswith("\n") else ""
            lines[i] = f"{prefix}{new_datetime}{line_ending}"
            changed = True
            break

    if not changed:
        return False

    new_text = "".join(lines) + "".join(lines[len(lines) :])
    try:
        Path(filepath).write_text(new_text, encoding="utf-8")
        return True
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: exception in header check: {e}")
        return False


def _get_repo_root() -> Path:
    """Git 仓库根目录（支持独立项目 commit-hooks 或仓库内 scripts/hooks）"""
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        cwd=os.getcwd(),
    )
    if r.returncode != 0:
        raise RuntimeError("无法获取 Git 仓库根目录，请确保在 Git 仓库内执行")
    return Path(r.stdout.strip())


def _load_llm_provider_config() -> dict:
    """加载 LLM 配置（支持 COMMIT_HOOKS_LLM_CONFIG 或 automanus.llm.toml / .commit-hooks.llm.toml）。"""
    try:
        import tomllib  # py>=3.11
    except ImportError:
        raise RuntimeError("tomllib 不可用（需要 Python 3.11+）")

    repo_root = _get_repo_root()
    cfg_env = os.environ.get("COMMIT_HOOKS_LLM_CONFIG")
    if cfg_env:
        cfg_path = Path(cfg_env) if Path(cfg_env).is_absolute() else repo_root / cfg_env
    else:
        for name in ("automanus.llm.toml", ".commit-hooks.llm.toml"):
            p = repo_root / name
            if p.exists():
                cfg_path = p
                break
        else:
            cfg_path = repo_root / "automanus.llm.toml"
    if not cfg_path.exists():
        raise RuntimeError(
            f"LLM 配置文件不存在: {cfg_path}\n"
            "可设置 COMMIT_HOOKS_LLM_CONFIG 或在仓库根创建 automanus.llm.toml / .commit-hooks.llm.toml"
        )

    with open(cfg_path, "rb") as f:
        config = tomllib.load(f)

    provider_name = (config.get("global", {}) or {}).get("llm_provider")
    if not provider_name:
        raise RuntimeError(
            f"LLM 配置缺失：请在 {cfg_path.name} 的 [global] 中设置 llm_provider"
        )

    provider_cfg = config.get(provider_name, {}) or {}
    if not provider_cfg:
        available = [k for k in config.keys() if k != "global"]
        raise RuntimeError(
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
                raise RuntimeError(
                    f"API Key 环境变量未设置：{api_key_env}\n"
                    f'例如：export {api_key_env}="your-api-key"'
                )
        else:
            raise RuntimeError(
                f"提供商 {provider_name} 缺少 api_key 或 api_key_env 配置"
            )

    if not base_url or not model:
        raise RuntimeError(
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


async def _async_llm_fix_last_updated(filepath: str, cfg: dict) -> bool:
    """使用 LLM 修复文件头中的"最后更新日期"字段（当自动修复失败时使用）。

    流程（参考 llm_fix_shellcheck.py）：
    1. LLM 返回修复后的文件头片段
    2. 本地精确替换文件头部分
    3. 使用 git diff --no-index 生成 patch
    4. 用 git apply --check 验证
    5. 用 git apply 应用

    返回:
        True: 成功修复
        False: 修复失败
    """
    import httpx
    import tempfile

    try:
        # 统一转换为绝对路径
        abs_filepath = Path(filepath).resolve()
        old_content = abs_filepath.read_text(encoding="utf-8")
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: exception in header check: {e}")
        return False

    # 确定文件头范围（前 30 行，足够覆盖所有文件头格式）
    lines = old_content.splitlines(keepends=True)
    header_end_idx = min(30, len(lines))

    # 提取文件头片段（用于发送给 LLM）
    header_snippet = "".join(lines[:header_end_idx])

    current_dt = _get_current_datetime_str()
    filename = abs_filepath.name
    # 计算相对于仓库根目录的路径（用于 patch 中的路径）
    repo_root = Path(__file__).resolve().parents[3]
    try:
        rel_path = str(abs_filepath.relative_to(repo_root))
    except ValueError:
        # 如果无法计算相对路径，使用原始文件路径
        rel_path = filepath

    prompt = f"""你是一个代码助手，任务是修复文件头中的"最后更新日期"字段。

## 任务
文件 `{filename}` 的文件头中"最后更新日期"字段需要更新为：`{current_dt}`

## 强制约束
1. **只修改"最后更新日期"字段**，禁止修改其他任何内容（包括创建日期、描述、文件名等）。
2. 如果文件头中**已有"最后更新日期"字段**，只更新其日期值。
3. 如果文件头中**缺少"最后更新日期"字段**，在"创建日期"字段之后添加一行，格式与"创建日期"保持一致（注释前缀、冒号风格等）。
4. **必须返回完整的文件头片段**（从文件开始到文件头注释结束），保持原有格式和行数，不要添加或删除行。

## 文件头片段（需要修复的部分）

```
{header_snippet}
```

请直接返回修复后的完整文件头片段（保持原有行数和格式），不要包含正文代码，不要使用 Markdown 代码块。"""

    api_type = cfg["api_type"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    timeout = cfg["timeout"]

    try:
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
                "max_tokens": 600,
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
                "max_tokens": 600,
                "temperature": 0.2,
            }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(api_url, headers=headers, json=body)
            resp.raise_for_status()
            result = resp.json()

        # 解析响应
        if "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message") or {}
            fixed_header = (msg.get("content") or msg.get("text") or "").strip()
        elif "content" in result and result["content"]:
            fixed_header = ""
            for block in result["content"]:
                if isinstance(block, str):
                    fixed_header = block.strip()
                elif isinstance(block, dict):
                    t = (block.get("text") or block.get("content") or "").strip()
                    if t:
                        fixed_header = t
        else:
            return False

        # 去掉可能的代码块标记
        if fixed_header.startswith("```"):
            lines_fixed = fixed_header.splitlines()
            out_lines = []
            in_block = False
            for line in lines_fixed:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.startswith("```"):
                    out_lines.append(line)
            fixed_header = "\n".join(out_lines).strip()

        if not fixed_header:
            return False

        # 确保 fixed_header 有换行符（splitlines(keepends=True) 需要）
        fixed_header_lines = fixed_header.splitlines(keepends=True)
        if fixed_header_lines and not fixed_header_lines[-1].endswith("\n"):
            fixed_header_lines[-1] = fixed_header_lines[-1] + "\n"
        fixed_header = "".join(fixed_header_lines)

        # 构建新文件内容：修复后的文件头 + 原文件正文
        new_content = fixed_header + "".join(lines[header_end_idx:])

        if new_content == old_content:
            # 未产生变更
            return False

        # 使用 git diff --no-index 生成 patch（参考 llm_fix_shellcheck.py）
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "old"
            new_file = Path(tmpdir) / "new"
            old_file.write_text(old_content, encoding="utf-8")
            new_file.write_text(new_content, encoding="utf-8")

            # git diff --no-index 生成 unified diff
            diff_result = subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--",
                    str(old_file),
                    str(new_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if diff_result.returncode not in (0, 1):
                # git diff --no-index 在文件不同时返回 1（正常），其他错误码才异常
                return False

            patch_text = diff_result.stdout
            if not patch_text.strip():
                # 无差异
                return False

            # 将临时文件路径替换为实际文件路径
            patch_lines = patch_text.splitlines(keepends=True)
            result_lines = []
            for line in patch_lines:
                if line.startswith("diff --git"):
                    result_lines.append(f"diff --git a/{rel_path} b/{rel_path}\n")
                elif line.startswith("--- a/") or line.startswith("--- /"):
                    result_lines.append(f"--- a/{rel_path}\n")
                elif line.startswith("+++ b/") or line.startswith("+++ /"):
                    result_lines.append(f"+++ b/{rel_path}\n")
                else:
                    result_lines.append(line)
            final_patch = "".join(result_lines)

            # 验证 patch（git apply --check）
            git_dir_result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                check=False,
            )
            git_dir = Path((git_dir_result.stdout or "").strip())
            if not git_dir:
                return False

            tmp_patch = git_dir / "llm_fix_header.patch"
            try:
                tmp_patch.write_text(final_patch, encoding="utf-8")
                check_result = subprocess.run(
                    ["git", "apply", "--check", str(tmp_patch)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if check_result.returncode != 0:
                    return False

                # 应用 patch
                apply_result = subprocess.run(
                    ["git", "apply", str(tmp_patch)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if apply_result.returncode != 0:
                    return False

                return True
            finally:
                try:
                    tmp_patch.unlink()
                except Exception as e:
                    if os.environ.get("DEBUG"):
                        print(f"DEBUG: failed to unlink patch: {e}")

    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: exception in header check: {e}")
        return False


def _llm_fix_last_updated(filepath: str) -> bool:
    """同步包装器：调用异步 LLM 修复函数。"""
    try:
        cfg = _load_llm_provider_config()
        return asyncio.run(_async_llm_fix_last_updated(filepath, cfg))
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: exception in header check: {e}")
        return False


def main() -> int:
    """主函数

    模式说明：
    - 默认（无参数）: 作为 pre-commit 钩子，仅做检测，不修改文件。
    - --fix-last-updated: 仅针对已暂存的“已修改文件”，自动更新文件头中的“最后更新日期”字段。
      * 只处理存在规范文件头且 diff 中缺失“最后更新日期”变更的文件
      * 不为缺失文件头的新增文件补充头部（保持由用户或其他工具处理）
    """
    parser = argparse.ArgumentParser(
        description="源码文件头检测与修复工具（多语言）",
        add_help=True,
    )
    parser.add_argument(
        "--fix-last-updated",
        action="store_true",
        help="仅针对已暂存的已修改文件，自动更新文件头中的“最后更新日期”字段",
    )
    args = parser.parse_args()

    # 设置 Windows 控制台编码为 UTF-8
    if sys.platform == "win32":
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    # 获取暂存区文件列表（新增 / 修改）
    added_files = get_staged_source_files("A")
    modified_files = get_staged_source_files("M")

    if not added_files and not modified_files:
        if args.fix_last_updated:
            print(f"{GREEN}[header-fix]{NC} 无需要修复的文件（暂存区为空）")
            return 0
        else:
            print(f"{GREEN}[pre-commit]{NC} 无需要检查的文件，跳过文件头检测")
            return 0

    # 修复模式：仅修复已修改文件的“最后更新日期”字段
    if args.fix_last_updated:
        targets: list[str] = []
        for filepath in modified_files:
            if filepath.endswith("__init__.py"):
                continue
            # 仅当存在“最后更新日期未更新”的违规时才尝试修复
            valid, errors = check_last_update_modified(filepath)
            if not valid and any("最后更新日期" in err for err in (errors or [])):
                targets.append(filepath)

        if not targets:
            print(f"{GREEN}[header-fix]{NC} 未发现需要修复“最后更新日期”的文件")
            return 0

        current_dt = _get_current_datetime_str()
        fixed = []
        failed = []
        for fp in targets:
            if _update_last_updated_field(fp, current_dt):
                fixed.append(fp)
            else:
                failed.append(fp)

        if fixed:
            # 将已修复文件重新加入暂存区，确保后续基于 --cached 的检查看到最新头部
            try:
                subprocess.run(
                    ["git", "add", "--"] + fixed,
                    text=True,
                    check=False,
                )
            except Exception as e:
                # git add 失败时仍然打印修复结果，但后续检查可能继续报错
                if os.environ.get("DEBUG"):
                    print(f"DEBUG: git add failed: {e}")

            print(f"{GREEN}[header-fix]{NC} 已自动更新以下文件的“最后更新日期”：")
            for fp in fixed:
                print(f"  - {fp}")

        if failed:
            print(f"{YELLOW}[header-fix]{NC} 以下文件未能自动修复：")
            for fp in failed:
                print(f"  - {fp}")
            print()

            # 检测是否有 TTY，如果有则询问是否使用 LLM 修复
            try:
                tty = open("/dev/tty", "r")
                tty.close()
                has_tty = True
            except (OSError, FileNotFoundError):
                has_tty = False

            if has_tty:
                print(
                    f"{YELLOW}是否使用 LLM 修复这些文件的“最后更新日期”字段?{NC} [Y/n] ",
                    end="",
                    flush=True,
                )
                try:
                    with open("/dev/tty", "r") as tty:
                        response = tty.readline().strip().lower()
                    if response not in ("n", "no"):
                        # 使用 LLM 修复
                        llm_fixed = []
                        llm_failed = []
                        for fp in failed:
                            print(
                                f"{YELLOW}[header-fix]{NC} 正在使用 LLM 修复: {fp}..."
                            )
                            if _llm_fix_last_updated(fp):
                                llm_fixed.append(fp)
                                # 修复成功后立即 git add
                                try:
                                    subprocess.run(
                                        ["git", "add", "--", fp],
                                        text=True,
                                        check=False,
                                    )
                                except Exception as e:
                                    if os.environ.get("DEBUG"):
                                        print(f"DEBUG: git add new file failed: {e}")
                            else:
                                llm_failed.append(fp)

                        if llm_fixed:
                            print(
                                f"{GREEN}[header-fix]{NC} LLM 已修复以下文件的“最后更新日期”："
                            )
                            for fp in llm_fixed:
                                print(f"  - {fp}")

                        if llm_failed:
                            print(
                                f"{RED}[header-fix]{NC} LLM 未能修复以下文件，请手动处理："
                            )
                            for fp in llm_failed:
                                print(f"  - {fp}")
                            return 1

                        # 所有文件都已修复
                        print(
                            f"{GREEN}[header-fix]{NC} 修复完成：建议重新运行 git diff / pre-commit 确认"
                        )
                        return 0
                    else:
                        # 用户选择不使用 LLM 修复
                        print(
                            f"{RED}[header-fix]{NC} 已跳过 LLM 修复，请手动更新文件头"
                        )
                        return 1
                except Exception:
                    # TTY 读取失败，回退到手动修复提示
                    if os.environ.get("DEBUG"):
                        print(f"DEBUG: TTY read failed: {sys.exc_info()[1]}")

            # 无 TTY 或用户选择不修复：提示手动修复
            print(
                f"{RED}[header-fix]{NC} 请手动更新文件头，或在项目内添加 llm_generate_header 等工具辅助生成"
            )
            return 1

        print(
            f"{GREEN}[header-fix]{NC} 修复完成：建议重新运行 git diff / pre-commit 确认"
        )
        return 0

    violations: list[tuple[str, list[str]]] = []
    new_file_violations: list[tuple[str, list[str]]] = []
    last_updated_violations: list[tuple[str, list[str]]] = []

    # 新增文件：检查完整文件头（类型 2：新建文件没有文件头 / 字段缺失）
    for filepath in added_files:
        if filepath.endswith("__init__.py"):
            continue
        valid, errors = check_file_header(filepath)
        if not valid:
            violations.append((filepath, errors))
            new_file_violations.append((filepath, errors))

    # 已修改文件：检查是否更新了最后更新日期（类型 1：最后更新日期未更新）
    for filepath in modified_files:
        if filepath.endswith("__init__.py"):
            continue
        valid, errors = check_last_update_modified(filepath)
        if not valid:
            violations.append((filepath, errors))
            last_updated_violations.append((filepath, errors))

    if violations:
        print(f"{RED}[pre-commit] 错误: 文件头不符合规范{NC}")
        print()
        for filepath, errors in violations:
            print(f"  {filepath}:")
            for error in errors:
                print(f"    - {error}")
        print()

        # 分类统计三种情况
        total = len(violations)
        count_last = len(last_updated_violations)
        count_new = len(new_file_violations)
        print("问题分类统计:")
        print(f"  1) 仅“最后更新日期”未更新的文件: {count_last} 个")
        print(f"  2) 新建文件缺少规范文件头的文件: {count_new} 个")
        print(
            f"  3) 两类问题同时存在的场景：整体来看即同时存在以上 1) 与 2) 两种类型的文件"
        )
        print()

        # 若存在可自动修复的"最后更新日期未更新"问题，且有 TTY，则询问是否自动修复
        can_fix_last = bool(last_updated_violations)
        can_fix_new = bool(new_file_violations)
        interactive_disabled = os.environ.get(
            "AUTOMANUS_HEADER_NO_INTERACTIVE", ""
        ).lower() in ("1", "true", "yes")

        # 尝试获取 TTY
        tty = None
        if not interactive_disabled:
            try:
                # Git hook 环境下 stdin 非终端，这里显式从 /dev/tty 读
                tty = open("/dev/tty", "r")
            except OSError:
                tty = None

        # 跟踪是否实际进行了修复
        did_fix_last = False
        did_fix_new = False

        # 1) 先处理"最后更新日期未更新"的问题
        if can_fix_last and tty is not None:
            print(f'{NC}检测到 {count_last} 个仅"最后更新日期"未更新的已修改文件。{NC}')
            print(
                f'{NC}是否自动更新这些文件头中的"最后更新日期"字段? [Y/n] {NC}',
                end="",
                flush=True,
            )
            response = tty.readline().strip().lower()

            if response in ("", "y", "yes"):
                # 调用自身的修复模式
                env = dict(os.environ)
                env["AUTOMANUS_HEADER_NO_INTERACTIVE"] = "1"
                cmd = [sys.executable, __file__, "--fix-last-updated"]
                result = subprocess.run(cmd, text=True, env=env, check=False)
                if result.returncode != 0:
                    print(
                        f'{RED}[pre-commit]{NC} 自动更新"最后更新日期"失败，请根据上方输出手动修复'
                    )
                    if tty:
                        tty.close()
                    print(
                        f"{RED}提示：{NC}修复文件头后重新提交，或使用 git commit --no-verify 跳过所有检查"
                    )
                    return 1
                did_fix_last = True
                print()  # 换行分隔

        # 2) 再处理"新建文件缺少规范文件头"的问题
        if can_fix_new and tty is not None:
            new_files_list = [fp for fp, _ in new_file_violations]
            print(f"{NC}检测到 {count_new} 个新建文件缺少规范文件头。{NC}")
            print(
                f"{NC}是否使用 LLM 为这些文件生成文件头? [Y/n] {NC}",
                end="",
                flush=True,
            )
            response = tty.readline().strip().lower()

            if response in ("", "y", "yes"):
                # 调用项目内 llm_generate_header 工具（支持 tools/ 或 scripts/tools/）
                repo_root = _get_repo_root()
                llm_header_script = None
                for candidate in (
                    repo_root / "tools" / "llm_generate_header.py",
                    repo_root / "scripts" / "tools" / "llm_generate_header.py",
                ):
                    if candidate.exists():
                        llm_header_script = candidate
                        break
                if llm_header_script is None:
                    print(
                        f"{RED}[header-fix]{NC} 未找到项目内 llm_generate_header 工具（可放置于 tools/ 或 scripts/tools/）"
                    )
                else:
                    cmd = [sys.executable, str(llm_header_script)] + new_files_list
                    print(f"{YELLOW}[header-fix]{NC} 正在使用 LLM 生成文件头...")
                    result = subprocess.run(cmd, text=True, check=False)
                    if result.returncode == 0:
                        # 生成成功后 git add 这些文件
                        for fp in new_files_list:
                            try:
                                subprocess.run(
                                    ["git", "add", "--", fp],
                                    text=True,
                                    check=False,
                                )
                            except Exception as e:
                                if os.environ.get("DEBUG"):
                                    print(f"DEBUG: git add failed: {e}")
                        did_fix_new = True
                        print()  # 换行分隔
                    else:
                        print(f"{RED}[header-fix]{NC} LLM 生成文件头失败，请手动处理")
                        if tty:
                            tty.close()
                        print(
                            f"{RED}提示：{NC}修复文件头后重新提交，或使用 git commit --no-verify 跳过所有检查"
                        )
                        return 1

        # 关闭 TTY
        if tty:
            tty.close()

        # 如果实际进行了任何修复，重新检查一次
        if (did_fix_last or did_fix_new) and not interactive_disabled:
            env = dict(os.environ)
            env["AUTOMANUS_HEADER_NO_INTERACTIVE"] = "1"
            recheck = subprocess.run(
                [sys.executable, __file__],
                text=True,
                env=env,
                check=False,
            )
            return recheck.returncode

        print("文件头模板示例（按语言区分）:")
        for lang, tpl in HEADER_TEMPLATES.items():
            print(f"[{lang}]")
            print(tpl)
            print()
        print(
            f"{RED}提示：{NC}修复文件头后重新提交，或使用 git commit --no-verify 跳过所有检查"
        )
        return 1

    print(f"{GREEN}[pre-commit]{NC} 文件头检测通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
