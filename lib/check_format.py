#!/usr/bin/env python3
"""
文件名: check_format.py
描述: Python 代码格式化检查（black）
创建日期: 2026年01月26日 14:42:11
最后更新日期: 2026年01月29日 02:36:59
"""

import subprocess
import sys
import os
from pathlib import Path

# 颜色定义
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

INTERACTIVE_TRUE = {"1", "true", "yes", "y", "on"}
INTERACTIVE_FALSE = {"0", "false", "no", "n", "off"}


def _env_true(key: str) -> bool:
    v = os.environ.get(key, "")
    return v.strip().lower() in INTERACTIVE_TRUE


def _env_false(key: str) -> bool:
    v = os.environ.get(key, "")
    return v.strip().lower() in INTERACTIVE_FALSE


def _is_interactive() -> bool:
    # black 自动修复的交互确认：默认开启（只要有 TTY）
    # 显式关闭：AUTOMANUS_BLACK_AUTOFIX=0/false/no
    if _env_false("AUTOMANUS_BLACK_AUTOFIX"):
        return False
    if _env_true("AUTOMANUS_BLACK_AUTOFIX"):
        return True
    try:
        return os.path.exists("/dev/tty")
    except Exception:
        return False


def _prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    """从 /dev/tty 进行交互式确认，避免 stdin 被 git 占用导致卡死。

    - 非交互环境（无 TTY）直接返回 False
    - 注意：字符串中使用英文 '?'，避免某些 shell 参数解析对中文标点不友好
    """
    try:
        try:
            with open("/dev/tty", "r", encoding="utf-8", errors="ignore") as tty_in:
                prompt = " [Y/n] " if default_yes else " [y/N] "
                sys.stdout.write(question + prompt)
                sys.stdout.flush()
                resp = tty_in.readline().strip().lower()
        except Exception:
            return False

        if resp == "":
            return default_yes
        return resp in {"y", "yes"}
    except Exception:
        return False


def get_staged_python_files() -> list[str]:
    """获取暂存区中的 Python 文件"""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACM",
        ],
        capture_output=True,
        text=True,
    )
    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith(".py") and f]


def _black_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "black", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_black_check(files: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "black", "--check", "--diff"] + files,
        capture_output=True,
        text=True,
    )


def _run_black_fix(files: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "black"] + files,
        capture_output=True,
        text=True,
    )


def _git_add(files: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "add", "--"] + files,
        capture_output=True,
        text=True,
    )


def main() -> int:
    """主函数"""
    files = get_staged_python_files()
    if not files:
        print(f"{GREEN}[pre-commit]{NC} 无修改的 Python 文件，跳过格式化检查")
        return 0

    # 检查 black 是否可用（使用当前 Python 解释器）
    if not _black_available():
        print(f"{YELLOW}[pre-commit]{NC} black 未安装，跳过格式化检查")
        print(f"{YELLOW}提示：{NC}安装 black 以启用检查: pip install black")
        return 0

    # 检查格式化（使用当前 Python 解释器）
    result = _run_black_check(files)

    if result.returncode != 0:
        # 交互模式：允许一键修复 + 重新检查 + 自动加入暂存区
        if _is_interactive():
            print(f"{YELLOW}[pre-commit]{NC} 检测到 black 不符合规范")
            print()
            print("需要格式化的文件:")
            for file in files:
                print(f"  - {file}")
            print()
            ok = _prompt_yes_no(
                "是否自动运行 black 修复并加入暂存区?", default_yes=True
            )
            if ok:
                fix = _run_black_fix(files)
                if fix.returncode != 0:
                    print(f"{RED}[pre-commit] 错误: 自动运行 black 失败{NC}")
                    if fix.stdout.strip():
                        print(fix.stdout.strip())
                    if fix.stderr.strip():
                        print(fix.stderr.strip())
                    return 1

                add = _git_add(files)
                if add.returncode != 0:
                    print(f"{RED}[pre-commit] 错误: git add 失败{NC}")
                    if add.stdout.strip():
                        print(add.stdout.strip())
                    if add.stderr.strip():
                        print(add.stderr.strip())
                    return 1

                recheck = _run_black_check(files)
                if recheck.returncode != 0:
                    print(f"{RED}[pre-commit] 错误: black 修复后仍未通过检查{NC}")
                    return 1

                print(
                    f"{GREEN}[pre-commit]{NC} 已自动运行 black 并加入暂存区，格式检查通过"
                )
                return 0

        # 只输出简要信息，避免在 commit 时刷屏
        print(f"{RED}[pre-commit] 错误: Python 代码格式不符合 black 规范{NC}")
        print()
        print("需要格式化的文件:")
        for file in files:
            print(f"  - {file}")
        print()
        print(f"{YELLOW}提示：{NC}运行以下命令自动修复:")
        print(f"  {sys.executable} -m black {' '.join(files)}")
        print(f"  然后重新提交: git add {' '.join(files)} && git commit")
        return 1

    print(f"{GREEN}[pre-commit]{NC} Python 代码格式检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
