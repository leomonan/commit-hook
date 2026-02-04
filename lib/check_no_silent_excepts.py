"""
文件名: check_no_silent_excepts.py
描述: 防回归检查 - 禁止在改动文件中引入“except: pass / except Exception: pass”这类静默吞异常代码
创建日期: 2025年12月18日 14:26:39
最后更新日期: 2026年02月01日 23:58:57
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    message: str


def _run_git(args: Sequence[str]) -> Tuple[int, str]:
    p = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.returncode, (p.stdout or "")


def _read_python_source(path: Path, *, staged: bool) -> str:
    """
    读取 Python 源码。

    - staged=True: 读取暂存区（index）版本，避免“工作区内容与暂存区不一致”导致漏检/误检
    - staged=False: 读取工作区文件
    """
    if not staged:
        return path.read_text(encoding="utf-8")

    code, out = _run_git(["show", f":{path.as_posix()}"])
    if code != 0:
        raise RuntimeError(out.strip() or "git show 失败")
    return out


def _is_shallow_repo() -> bool:
    code, out = _run_git(["rev-parse", "--is-shallow-repository"])
    return code == 0 and out.strip().lower() == "true"


def _ensure_base_ref_available(base_ref: str) -> None:
    code, _out = _run_git(["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
    if code == 0:
        return

    if not base_ref.startswith("origin/"):
        return

    branch = base_ref.split("/", 1)[1].strip()
    if not branch:
        return

    fetch_refspec = f"{branch}:refs/remotes/origin/{branch}"
    print(
        f"ℹ️  check_no_silent_excepts: 基准引用 {base_ref} 不存在，尝试执行 git fetch origin {fetch_refspec}",
        file=sys.stderr,
    )
    _run_git(["fetch", "origin", fetch_refspec, "--no-tags", "--prune"])


def _changed_python_files(base_ref: str) -> List[str]:
    """
    只检查“本分支相对base_ref”的改动文件，避免一次性要求清理全仓库历史遗留。
    """
    _ensure_base_ref_available(base_ref)
    code, out = _run_git(
        ["diff", "--name-only", "--diff-filter=ACMRT", f"{base_ref}...HEAD"]
    )
    if code != 0:
        hint = ""
        out_lc = out.lower()
        if (
            "no merge base" in out_lc
            or "unknown revision" in out_lc
            or "bad revision" in out_lc
        ):
            if _is_shallow_repo():
                hint = "\n提示：当前仓库为 shallow clone，需拉全历史/远端引用（CI 建议 actions/checkout 设置 fetch-depth: 0）。"
            else:
                hint = "\n提示：请确认已拉取远端分支引用（例如 origin/main），且与当前分支存在共同祖先。"
        raise RuntimeError(
            f"无法获取 git diff 文件列表（base_ref={base_ref}）\n{out}{hint}"
        )
    files = []
    for line in out.splitlines():
        f = line.strip()
        if f.endswith(".py") and f:
            files.append(f)
    return files


def _iter_findings_for_file(path: Path, *, staged: bool) -> Iterable[Finding]:
    try:
        src = _read_python_source(path, staged=staged)
    except Exception as e:
        if staged:
            yield Finding(str(path), 1, "read_error", f"读取暂存区内容失败: {e}")
        else:
            yield Finding(str(path), 1, "read_error", f"读取文件失败: {e}")
        return

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        yield Finding(
            str(path),
            int(getattr(e, "lineno", 1) or 1),
            "syntax_error",
            f"语法错误: {e}",
        )
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            if not isinstance(h, ast.ExceptHandler):
                continue

            # 仅禁止“except ...: pass”这类纯静默吞异常
            if len(h.body) != 1 or not isinstance(h.body[0], ast.Pass):
                continue

            # except:
            if h.type is None:
                yield Finding(
                    str(path),
                    int(getattr(h, "lineno", 1) or 1),
                    "except-pass",
                    "禁止使用“except: pass”（静默吞异常）",
                )
                continue

            # except Exception:
            if isinstance(h.type, ast.Name) and h.type.id == "Exception":
                yield Finding(
                    str(path),
                    int(getattr(h, "lineno", 1) or 1),
                    "except-exception-pass",
                    "禁止使用“except Exception: pass”（静默吞异常）",
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="禁止在改动文件中引入静默吞异常(except-pass)"
    )
    parser.add_argument(
        "--base", default="origin/main", help="对比基准分支/引用（默认origin/main）"
    )
    parser.add_argument(
        "--files", nargs="*", help="显式指定要检查的文件列表（覆盖--base自动diff）"
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="检查暂存区（index）版本，而不是工作区文件内容",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.files:
        files = [f for f in args.files if f and f.endswith(".py")]
    else:
        files = _changed_python_files(args.base)

    findings: List[Finding] = []
    for f in files:
        p = Path(f)
        if not p.exists() or not p.is_file():
            continue
        findings.extend(list(_iter_findings_for_file(p, staged=bool(args.staged))))

    if not findings:
        print("✅ check_no_silent_excepts: 通过")
        return 0

    print("❌ check_no_silent_excepts: 发现静默吞异常代码，请修复后再提交：")
    for it in findings:
        print(f"- {it.file}:{it.line} [{it.kind}] {it.message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
