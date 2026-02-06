#!/usr/bin/env python3
"""
文件名: llm_generate_header.py
描述: 使用 LLM 为缺少文件头的新建文件生成规范文件头（显式命令，由 check_header.py 调用或独立运行）
创建日期: 2026年01月29日 17:42:20
最后更新日期: 2026年02月07日 00:40:00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from llm_client import (
    ConfigurationError,
    RED,
    GREEN,
    YELLOW,
    NC,
    repo_root,
    load_llm_config,
    call_llm,
    strip_code_fences,
)


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
        raise RuntimeError(f"读取文件失败: {path}: {e}")

    # 如果文件已经包含"文件名: "且位于前几行，则认为已有文件头，避免重复插入
    head = text[:400]
    if "文件名:" in head and "创建日期:" in head and "最后更新日期:" in head:
        raise RuntimeError(f"文件已存在文件头，不执行生成: {path}")

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
        cfg = load_llm_config()
    except ConfigurationError as e:
        print(f"{RED}[llm-header]{NC} 配置错误: {e}")
        return 1

    root = repo_root()
    ok_files: List[str] = []
    failed_files: List[str] = []

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
            raw = call_llm(
                prompt,
                cfg,
                max_tokens=400,
                temperature=0.2,
                log_prefix="llm-header",
            )
            header = strip_code_fences(raw)
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
