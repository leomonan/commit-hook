#!/usr/bin/env python3
"""
文件名: llm_review.py
描述: LLM 辅助代码审查模块
创建日期: 2026年01月25日 15:32:00
最后更新日期: 2026年02月01日 14:21:16
"""

import hashlib
import subprocess
import sys
import os
import json
import asyncio
import re
import time
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# 颜色定义
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

# 请求超时上限（秒），避免配置过大；实际使用 min(provider.timeout, REQUEST_TIMEOUT_CAP)
REQUEST_TIMEOUT_CAP = 180

# 缓存最大条目数（LRU 策略：超过此数量时删除最旧的条目）
CACHE_MAX_SIZE = 100

# 统计记录保留数量（最近 N 次审查的统计信息）
STATS_MAX_SIZE = 50

# 建议 commit message 存储文件（供 commit-msg hook 读取）
SUGGESTION_FILE_NAME = "llm_review_suggestion.txt"

# 检查项
CHECK_ITEMS = [
    "禁降级/禁兜底检测",
    "SSOT 违反检测",
    "单一路径策略检测",
    "LLM 与本地职责边界检测",
    "过度工程化检测",
    "产品长期化检测",
    "重复代码检测",
    "硬编码与魔法数字检测",
]

# 需要 LLM 审查的文件扩展名（SSOT：唯一审查规则来源）
# 规则：在列表里 → 审查；不在列表里 → 跳过（不做任何猜测或 fallback）
REVIEW_EXTENSIONS = {
    # Python
    ".py",
    ".pyw",
    ".pyi",
    # JavaScript/TypeScript
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ksh",
    # Ruby
    ".rb",
    ".rake",
    # PHP
    ".php",
    ".phtml",
    # Go
    ".go",
    # Rust
    ".rs",
    # JVM
    ".java",
    ".kt",
    ".scala",
    ".groovy",
    # C/C++
    ".cpp",
    ".cxx",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".hxx",
    # C#
    ".cs",
    # Swift
    ".swift",
    # Objective-C
    ".m",
    ".mm",
    # Lua
    ".lua",
    # Perl
    ".pl",
    ".pm",
    # R
    ".r",
    ".R",
    # SQL
    ".sql",
    # 配置文件（明确需要审查的）
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".xml",
    # 前端框架
    ".vue",
    ".svelte",
    # Dart
    ".dart",
    # Clojure
    ".clj",
    ".cljs",
    ".cljc",
    # Elixir
    ".ex",
    ".exs",
    # Erlang
    ".erl",
    ".hrl",
    # Haskell
    ".hs",
    ".lhs",
    # OCaml
    ".ml",
    ".mli",
    # F#
    ".fs",
    ".fsx",
    ".fsi",
    # Julia
    ".jl",
    # Nim
    ".nim",
    # Zig
    ".zig",
    # Verilog/SystemVerilog
    ".v",
    ".sv",
    # TCL
    ".tcl",
    # PowerShell
    ".ps1",
    ".psm1",
    # Windows Batch
    ".bat",
    ".cmd",
    # Makefile
    ".makefile",
    ".mk",
}


def should_review_file(filepath: str) -> bool:
    """判断文件是否需要 LLM 审查（SSOT）

    规则（唯一真相来源）：
    - 在 REVIEW_EXTENSIONS 列表里 → 审查
    - 无扩展名的特殊文件在 SPECIAL_FILES 列表里 → 审查
    - 不在任何列表里 → 跳过（不做任何猜测或 fallback）

    这是唯一的审查规则来源，符合 SSOT/KISS/禁兜底原则。
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    # 1. 检查扩展名
    if ext in REVIEW_EXTENSIONS:
        return True

    # 2. 检查无扩展名的特殊文件（明确列出需要审查的）
    filename_lower = path.name.lower()
    SPECIAL_FILES = {
        "makefile",
        "dockerfile",
        "rakefile",
        "gemfile",
        "procfile",
        ".gitignore",
        ".dockerignore",
        ".npmignore",
        ".editorconfig",
        ".prettierrc",
        ".eslintrc",
    }
    if filename_lower in SPECIAL_FILES:
        return True

    # 3. 不在列表里 → 不审查（明确失败，不做猜测）
    return False


def _parse_diff_and_filter(diff: str) -> Tuple[str, list]:
    """解析 git diff，过滤出需要审查的文件

    规则（SSOT）：
    - should_review_file() 返回 True → 保留完整 diff
    - should_review_file() 返回 False → 跳过，记录文件名

    Returns:
        (filtered_diff, skipped_files):
        - filtered_diff: 需要审查的文件的完整 diff
        - skipped_files: 跳过的文件列表，格式为 [(status, filepath), ...]
    """
    if not diff.strip():
        return "", []

    lines = diff.split("\n")
    filtered_lines = []
    skipped_files = []
    current_file = None
    should_review = None  # None=未确定, True=需要审查, False=跳过
    current_status = None  # 'A' (新增), 'M' (修改), 'D' (删除)

    i = 0
    while i < len(lines):
        line = lines[i]

        # 匹配 diff --git 行：diff --git a/path b/path
        diff_match = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
        if diff_match:
            # 保存上一个文件的状态
            if current_file is not None and should_review is False:
                if current_status:
                    skipped_files.append((current_status, current_file))

            # 解析新文件
            file_a = diff_match.group(1)
            file_b = diff_match.group(2)

            # 判断文件状态
            if file_b == "/dev/null":
                current_status = "D"  # 删除
                current_file = file_a
            elif file_a == "/dev/null":
                current_status = "A"  # 新增
                current_file = file_b
            else:
                current_status = "M"  # 修改
                current_file = file_b

            # 判断是否需要审查（SSOT）
            should_review = should_review_file(current_file)

            if should_review:
                # 需要审查：保留 diff --git 行
                filtered_lines.append(line)
            # else: 跳过，不保留任何内容

        # 处理 diff 内容行
        elif current_file is not None:
            if should_review:
                # 需要审查：保留所有行
                filtered_lines.append(line)
            # else: 跳过所有 diff 内容行

        i += 1

    # 处理最后一个文件
    if current_file is not None and should_review is False and current_status:
        skipped_files.append((current_status, current_file))

    # 构建过滤后的 diff
    filtered_diff = "\n".join(filtered_lines)

    # 如果有跳过的文件，添加说明
    if skipped_files:
        skipped_section = ["\n# 跳过审查的文件（不在审查列表中）："]
        for status, filepath in skipped_files:
            status_name = {"A": "新增", "M": "修改", "D": "删除"}.get(status, "变更")
            skipped_section.append(f"# {status_name}: {filepath}")
        filtered_diff += "\n" + "\n".join(skipped_section)

    return filtered_diff, skipped_files


class ConfigurationError(Exception):
    """配置错误异常（应阻断提交）"""

    pass


class NetworkError(Exception):
    """网络/依赖/API 等运行时错误；main 中会阻断提交并提示跳过方式"""

    pass


def get_staged_diff() -> str:
    """获取暂存区的 diff 内容（已过滤：不在审查列表中的文件会被跳过）"""
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # 替换无法解码的字符
    )
    raw_diff = result.stdout

    # 解析并过滤 diff：只保留需要审查的文件
    filtered_diff, _ = _parse_diff_and_filter(raw_diff)
    return filtered_diff


def _format_raw_response(obj, max_len: int = 4000) -> str:
    """将 API 响应对象格式化为可读字符串，用于错误信息。"""
    try:
        raw = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        raw = repr(obj)
    return raw[:max_len] + ("\n... (已截断)" if len(raw) > max_len else "")


def _get_repo_root() -> Path:
    """Git 仓库根目录（支持独立项目或仓库内 commit-hooks）"""
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        cwd=os.getcwd(),
    )
    if r.returncode != 0:
        raise ConfigurationError("无法获取 Git 仓库根目录，请确保在 Git 仓库内执行")
    return Path(r.stdout.strip())


def _cache_dir() -> Path:
    """LLM Review 缓存目录；可通过环境变量 COMMIT_HOOKS_CACHE_DIR 覆盖"""
    if os.environ.get("COMMIT_HOOKS_CACHE_DIR"):
        return Path(os.environ["COMMIT_HOOKS_CACHE_DIR"])
    return Path.home() / ".cache" / "commit-hooks" / "llm_review"


def _load_cache() -> dict:
    """从磁盘加载缓存：{ hash -> {summary, issues, commit_message?} }"""
    path = _cache_dir() / "cache.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    """将缓存写入磁盘。失败则抛出异常，不静默吞掉。

    如果缓存条目数超过 CACHE_MAX_SIZE，使用 LRU 策略删除最旧的条目。
    """
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)

    # LRU 清理：如果超过最大大小，删除最旧的条目
    # 注意：由于 JSON 是无序的，我们使用简单的策略：保留最后 N 个条目
    if len(data) > CACHE_MAX_SIZE:
        # 将字典转换为列表，保留最后 CACHE_MAX_SIZE 个条目
        items = list(data.items())
        # 保留最后 CACHE_MAX_SIZE 个条目（最近使用的）
        data = dict(items[-CACHE_MAX_SIZE:])

    (d / "cache.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _load_stats() -> List[Dict]:
    """从磁盘加载统计记录：返回最近 N 次审查的统计信息列表

    每条记录格式：{"timestamp": 时间戳, "certain": 确定问题数, "uncertain": 不确定问题数, "total": 总问题数}
    """
    path = _cache_dir() / "stats.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 确保返回的是列表格式
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_stats(stats: List[Dict]) -> None:
    """将统计记录写入磁盘。失败则抛出异常，不静默吞掉。

    如果记录数超过 STATS_MAX_SIZE，保留最近 N 条。
    """
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)

    # 保留最近 STATS_MAX_SIZE 条记录
    if len(stats) > STATS_MAX_SIZE:
        stats = stats[-STATS_MAX_SIZE:]

    (d / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _record_stats(
    certain_count: int, uncertain_count: int, total_count: int, ignored: bool = False
) -> None:
    """记录本次审查的统计信息"""
    try:
        stats = _load_stats()
        stats.append(
            {
                "timestamp": time.time(),
                "certain": certain_count,
                "uncertain": uncertain_count,
                "total": total_count,
                "ignored": ignored,
            }
        )
        _save_stats(stats)
    except Exception as e:
        # 统计信息仅为辅助，失败不影响主流程
        if os.environ.get("DEBUG"):
            print(f"DEBUG: stats operation failed: {e}")


def _update_last_stats_ignored() -> None:
    """更新最近一条统计记录，标记为“已忽略不确定问题”"""
    try:
        stats = _load_stats()
        if not stats:
            return
        # 更新最后一条记录
        stats[-1]["ignored"] = True
        _save_stats(stats)
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"DEBUG: stats update failed: {e}")


def _print_stats_summary() -> None:
    """打印最近 N 次审查的统计摘要（仅在发现问题时显示）"""
    try:
        stats = _load_stats()
        if not stats:
            return

        # 计算最近 N 次（最多 20 次）的统计
        recent_stats = stats[-20:] if len(stats) > 20 else stats

        total_reviews = len(recent_stats)
        total_certain = sum(s.get("certain", 0) for s in recent_stats)
        total_uncertain = sum(s.get("uncertain", 0) for s in recent_stats)
        total_issues = sum(s.get("total", 0) for s in recent_stats)

        # 计算平均值
        avg_certain = total_certain / total_reviews if total_reviews > 0 else 0
        avg_uncertain = total_uncertain / total_reviews if total_reviews > 0 else 0
        avg_total = total_issues / total_reviews if total_reviews > 0 else 0

        # 计算有确定问题的审查次数占比
        reviews_with_certain = sum(1 for s in recent_stats if s.get("certain", 0) > 0)
        certain_rate = (
            (reviews_with_certain / total_reviews * 100) if total_reviews > 0 else 0
        )

        # 计算不确定问题的忽略率
        # 分母：存在不确定问题的审查次数
        reviews_with_uncertain = sum(
            1 for s in recent_stats if s.get("uncertain", 0) > 0
        )
        # 分子：存在不确定问题且被标记为忽略的次数
        reviews_ignored = sum(
            1
            for s in recent_stats
            if s.get("uncertain", 0) > 0 and s.get("ignored", False)
        )
        ignore_rate = (
            (reviews_ignored / reviews_with_uncertain * 100)
            if reviews_with_uncertain > 0
            else 0
        )

        print(f"{YELLOW}[统计]{NC} 最近 {total_reviews} 次审查：")
        print(
            f"  平均每次：{avg_total:.1f} 个问题（确定 {avg_certain:.1f}，不确定 {avg_uncertain:.1f}）"
        )
        print(
            f"  确定问题触发率：{certain_rate:.1f}% ({reviews_with_certain}/{total_reviews})"
        )
        if reviews_with_uncertain > 0:
            print(
                f"  不确定问题忽略率：{ignore_rate:.1f}% ({reviews_ignored}/{reviews_with_uncertain})"
            )
        print()
    except Exception as e:
        # 统计信息仅为辅助，失败不影响主流程
        if os.environ.get("DEBUG"):
            print(f"DEBUG: stats operation failed: {e}")


def _build_prompt(diff: str) -> str:
    """构造 LLM 审查用的 prompt，仅依赖 diff。"""
    return f"""你是一个代码审查助手。请严格按照以下项目设计原则分析 git diff。

## 审查原则（最高优先级）

你需要对每个潜在问题给出“确定性”判断：
- **certain（确定）**：你高度确信违反了某条原则，证据充分、规则匹配清晰
- **uncertain（不确定）**：存在一定风险或异味，但根据当前 diff 信息无法 100% 确认违反原则

## 核心原则（红线，不可违反）

1. **禁降级/禁兜底**
   - 违反：失败后静默继续执行、用默认值替代报错、catch 后不 raise 而是换一条路径
   - 正确：失败后 raise 异常或返回错误，让调用方知道失败了
   - 正确：在 raise 之前为错误信息生成辅助内容（如格式化、截断）不算降级

2. **SSOT（单一事实来源）**
   - 违反：同一配置在多处定义、多个真相来源、配置优先级冲突、双ID映射
   - 正确：每个概念只有一个权威定义

3. **单一路径策略**
   - 违反：为同一功能提供多种实现方式供选择、保留旧方案作为备选
   - 正确：正常的条件判断（参数校验、类型判断、格式解析）不算多路径
   - 正确：解析 API 响应时按字段规范提取内容不算多路径

4. **LLM 与本地职责边界**
   - 违反：本地代码做语义猜测、智能修补、自动纠错、用默认值补全 LLM 应返回的内容
   - 正确：LLM 负责决策和生成，本地仅做轨道控制（校验、报错）

## 设计原则

5. **过度工程化**
   - 违反：Enhanced/Advanced/V2 等暴露实现细节的命名、不必要的抽象层、预防性设计
   - 正确：KISS 原则，最简方案优先

6. **产品长期化**
   - 违反：代码注释中明确写有 TODO/FIXME 字样、"先这样后面再改"等临时方案标记
   - 正确：一次性卓越，不留技术债
   - 注意：只报告代码注释中字面包含 TODO/FIXME 的情况，不要推断

7. **重复代码**
   - 违反：明显可以复用但复制粘贴的逻辑（相似度极高的代码块）
   - 正确：DRY 原则

8. **硬编码与魔法数字**
   - 违反：业务逻辑中硬编码的配置值、无说明的魔法数字
   - 正确：配置化、规则化
   - 排除：配置文件（*.toml、*.yaml、*.env、*.ini、config.*）中的键值属于配置数据，不报告

## 审查排除

- **配置文件**：*.toml、*.yaml、*.env、*.ini 等中的 key-value 为正常配置，不报告
- **Markdown 文档**：*.md 文件不进行原则检查，不报告
- **正确的错误处理**：try-except 中 raise 了异常、或在 raise 前做格式化/日志，都是正确做法，不报告

**输出要求**：无论是否发现问题，**必须**返回 JSON，且 **summary 必填**。summary 为 1–2 句话的简短结论，会直接打印给用户：
- 发现问题时：说明发现了几类/几个问题，并区分确定/不确定（如「发现 3 个问题：2 个确定硬编码、1 个不确定重复代码」）
- 未发现问题时：用 1 句话概括本次 diff 做了什么，并说明符合原则（如「本次变更调整 LLM 提示词与错误处理，符合项目原则，未发现问题」）
- 禁止仅写「未发现潜在问题」等过简表述，summary 须含对本次变更的简要概括。

请以 JSON 格式返回，必须包含 summary 与 issues：
{{
    "summary": "1–2 句话结论（发现问题时写类型与数量，并区分确定/不确定；未发现时概括 diff 并说明符合原则）",
    "issues": [
        {{
            "file": "文件路径",
            "line": "行号范围",
            "type": "问题类型(fallback/ssot/single_path/llm_boundary/over_engineering/tech_debt/duplication/hardcode)",
            "certainty": "certain 或 uncertain（必须其一，不得省略）",
            "message": "问题描述（说明违反了哪条原则，为什么是问题，建议如何修复）"
        }}
    ]
}}

如果没有发现问题，必须额外返回 commit_message。格式：<type>(<scope>): <subject>
- type 取其一：feat, fix, refactor, docs, test, chore, perf, build, style, revert
- **重要**：commit_message 的 subject 部分必须使用中文描述
- 示例：feat(hooks): 添加 Git Commit Hooks 机制；fix(auth): 修复令牌过期问题
- 请根据 diff 内容生成一条符合格式的 commit_message（subject 使用中文）

无问题时的完整示例：
{{
    "summary": "本次变更调整 LLM 提示词与错误处理，符合项目原则，未发现问题",
    "issues": [],
    "commit_message": "fix(hooks): 强化 LLM Review 的 summary 与错误信息"
}}

以下是 git diff 内容：

**注意**：为了节省 tokens，不在审查列表中的文件（如 Markdown、图片、文档等）会被完全跳过，不会出现在 diff 中。只有明确需要审查的文件类型才会包含完整的 diff 内容。

```diff
{diff}
```

请仅返回 JSON，不要包含其他说明文字。"""


def call_llm_review(diff: str) -> Optional[dict]:
    """调用 LLM 进行代码审查"""
    try:
        # 使用 asyncio 运行异步函数
        return asyncio.run(_async_llm_review(diff))
    except (ConfigurationError, NetworkError):
        # 配置错误和网络错误直接向上传播
        raise
    except Exception as e:
        # 其他未知错误包装为网络错误
        raise NetworkError(f"LLM Review 调用失败: {e}")


async def _async_llm_review(diff: str) -> Optional[dict]:
    """异步调用 LLM 进行代码审查

    使用简化的实现，直接调用 httpx 而不依赖完整的 LLM 客户端
    遵循 SSOT 原则：review_provider 是 LLM Review 模型配置的唯一来源
    """
    try:
        # 1. 构造 prompt（仅依赖 diff），用于缓存 key 与 API 请求
        prompt = _build_prompt(diff)
        # 2. 查缓存：输入 hash 命中则直接返回，避免重复调用 LLM
        # 如果设置了 COMMIT_HOOKS_LLM_REVIEW_FORCE=1，则跳过缓存
        force_refresh = os.environ.get("COMMIT_HOOKS_LLM_REVIEW_FORCE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache = _load_cache()
        if not force_refresh and h in cache:
            print(f"{YELLOW}[LLM Review]{NC} 分析中... 命中缓存")
            print(
                f"{YELLOW}提示：{NC}使用 COMMIT_HOOKS_LLM_REVIEW_FORCE=1 git commit 可强制刷新（跳过缓存）"
            )
            return cache[h]
        if force_refresh:
            print(f"{YELLOW}[LLM Review]{NC} 分析中... 强制刷新（跳过缓存）")
        # 3. 未命中：加载配置并调用 API
        import httpx
        import tomllib

        # 读取 LLM 配置：只在钩子目录读取（或显式通过环境变量指定）
        hooks_root = Path(__file__).resolve().parents[1]
        cfg_env = os.environ.get("COMMIT_HOOKS_LLM_CONFIG")
        if cfg_env:
            llm_config_path = Path(cfg_env)
            if not llm_config_path.is_absolute():
                llm_config_path = hooks_root / cfg_env
        else:
            llm_config_path = hooks_root / "commit-hooks.llm.toml"

        if not llm_config_path.exists():
            raise ConfigurationError(
                f"LLM 配置文件不存在: {llm_config_path}\n"
                f"请在钩子目录创建 commit-hooks.llm.toml：{hooks_root}\n"
                "或设置环境变量 COMMIT_HOOKS_LLM_CONFIG 指定配置文件路径（绝对路径或相对钩子目录）"
            )

        with open(llm_config_path, "rb") as f:
            config = tomllib.load(f)

        # 读取 LLM Review 专用提供商（必需配置，遵循 SSOT 原则）
        provider_name = config.get("global", {}).get("review_provider")

        if not provider_name:
            raise ConfigurationError(
                "LLM Review 配置缺失\n"
                f"请在 {llm_config_path.name} 的 [global] 部分添加：\n"
                '  review_provider = "anthropic"  # 或其他提供商名称'
            )

        provider_config = config.get(provider_name, {})

        if not provider_config:
            available = [k for k in config.keys() if k != "global"]
            raise ConfigurationError(
                f"提供商 '{provider_name}' 配置不存在\n"
                f"可用提供商：{', '.join(available)}"
            )

        # 获取配置
        api_type = provider_config.get("api_type", "openai")
        base_url = provider_config.get("base_url")
        model = provider_config.get("model")
        timeout = provider_config.get("timeout", 60)
        request_timeout = min(timeout, REQUEST_TIMEOUT_CAP)

        # 处理 API Key（支持直接配置或环境变量）
        api_key = provider_config.get("api_key")
        if not api_key:
            api_key_env = provider_config.get("api_key_env")
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

        if not all([base_url, model]):
            raise ConfigurationError(
                f"提供商 {provider_name} 配置不完整（缺少 base_url 或 model）"
            )

        print(f"{YELLOW}[LLM Review]{NC} 分析中... 检视模型: {provider_name} / {model}")

        if os.environ.get("COMMIT_HOOKS_LLM_REVIEW_PRINT_PROMPT", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            print(f"{YELLOW}[LLM Review] 完整提示词：{NC}")
            print("---")
            print(prompt)
            print("---")

        # 根据 API 类型构建请求
        if api_type == "anthropic":
            # Anthropic API 格式
            api_url = f"{base_url}/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }

            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.3,
            }

            # 发送请求
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await asyncio.wait_for(
                    client.post(api_url, headers=headers, json=request_body),
                    timeout=request_timeout,
                )
                response.raise_for_status()

                result = response.json()
        else:
            # OpenAI 兼容格式（DeepSeek, Doubao, OpenAI 等）
            api_url = f"{base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            # 发送请求
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await asyncio.wait_for(
                    client.post(api_url, headers=headers, json=request_body),
                    timeout=request_timeout,
                )
                response.raise_for_status()

                result = response.json()

        # 统一解析：支持 OpenAI 格式与 Anthropic 格式（代理可能返回任一）
        if "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message") or {}
            content = (msg.get("content") or msg.get("text") or "").strip()
        elif "content" in result and result["content"]:
            # 遍历所有 content 块：Anthropic 扩展思考等格式下，首块可能为 type "thinking"（仅含 thinking 键），
            # 终答在后续 type "text" 的 text 中；取最后一个非空 text/content 作为 content
            content = ""
            for block in result["content"]:
                if isinstance(block, str):
                    content = block.strip()
                elif isinstance(block, dict):
                    block_text = (
                        block.get("text") or block.get("content") or ""
                    ).strip()
                    if block_text:
                        content = block_text
        else:
            _raw = _format_raw_response(result)
            raise NetworkError(
                f"API 响应格式异常：缺少 choices 或 content\n响应原文:\n{_raw}"
            )
        if not content:
            _raw = _format_raw_response(result)
            raise NetworkError(f"API 响应无有效文本内容\n响应原文:\n{_raw}")

        # 尝试提取 JSON（可能被包裹在 markdown 代码块中）
        if content.startswith("```"):
            # 移除 markdown 代码块标记
            lines = content.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.startswith("```")):
                    json_lines.append(line)
            content = "\n".join(json_lines).strip()

        # 解析 JSON 并写入缓存
        parsed = json.loads(content)
        c = _load_cache()
        # 如果 key 已存在，先删除（确保新条目在最后，符合 LRU）
        if h in c:
            del c[h]
        c[h] = parsed
        _save_cache(c)
        return parsed

    except asyncio.TimeoutError:
        raise NetworkError(f"LLM Review 超时（{request_timeout}秒）")
    except json.JSONDecodeError as e:
        raise NetworkError(f"LLM 返回的 JSON 格式错误: {e}")
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = (e.response.text or "").strip()
        except Exception:
            try:
                body = (
                    (e.response.content or b"")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                body = "（无法读取响应体）"
        if not body:
            body = "（无返回内容）"
        else:
            body = body[:2000] + ("..." if len(body) > 2000 else "")
        url_info = ""
        try:
            url_info = f"\n请求 URL: {e.request.url}"
        except Exception:
            url_info = ""
        raise NetworkError(
            f"HTTP 错误: {e.response.status_code}{url_info}\n返回内容:\n{body}"
        )
    except ConfigurationError:
        # 配置错误直接向上传播
        raise
    except Exception as e:
        # 其他错误视为网络错误
        raise NetworkError(f"LLM API 调用失败: {e}")


def _handle_error(tag: str, message: str, hint_extra: str = "") -> int:
    """统一处理配置/网络/未知错误：打印并阻断，返回 1。"""
    print(f"{RED}[LLM Review]{NC} {tag}:")
    if message:
        print(f"{RED}{message}{NC}")
    print()
    print(f"{RED}[LLM Review]{NC} 提交已阻断")
    base = "使用 COMMIT_HOOKS_NO_REVIEW=1 git commit 可跳过 LLM Review"
    hint = f"{hint_extra}；或{base}" if hint_extra else base
    print(f"{YELLOW}提示：{NC}{hint}")
    return 1


def _write_commit_suggestion(suggestion: str) -> None:
    """将 LLM 建议的 commit message 写入 .git 目录供 commit-msg 使用"""
    suggestion = (suggestion or "").strip()
    try:
        # 获取 Git 仓库 .git 目录（不做猜测，失败时直接返回）
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=False,
        )
        git_dir = (result.stdout or "").strip()
        if not git_dir:
            return
        path = Path(git_dir) / SUGGESTION_FILE_NAME
        if not suggestion:
            # 无建议时清理旧文件
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        path.write_text(suggestion + "\n", encoding="utf-8")
    except Exception:
        # 建议仅为辅助信息，写入失败不影响主流程
        return


def prompt_user_continue() -> bool:
    """提示用户是否继续提交

    默认为 n（不继续），用户需要明确输入 y 才能继续提交。
    Git 运行 hook 时 stdin 不接终端，故从 /dev/tty 读取；无 TTY 时默认不继续。
    """
    print()
    print(f"{YELLOW}请选择：{NC}输入 y 继续提交，输入 n 或直接回车 取消提交")
    print("继续提交? [y/N] ", end="", flush=True)
    try:
        with open("/dev/tty", "r") as tty:
            response = tty.readline().strip().lower()
        return response in ("y", "yes")
    except (EOFError, OSError, FileNotFoundError):
        # 无 TTY（Git hook、Windows、CI）：默认不继续
        print()
        return False


def main() -> int:
    """主函数"""
    # 设置 Windows 控制台编码为 UTF-8
    if sys.platform == "win32":
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    diff = get_staged_diff()
    if not diff:
        print(f"{GREEN}[LLM Review]{NC} 无变更内容，跳过")
        return 0

    try:
        result = call_llm_review(diff)
        summary = (result or {}).get("summary") or ""
        issues = (result or {}).get("issues") or []
        if not summary.strip():
            raise NetworkError("LLM 未返回 summary，审查结果无效")
    except ConfigurationError as e:
        return _handle_error("配置错误", str(e), "修复配置后重新提交")
    except NetworkError as e:
        return _handle_error("错误", str(e))
    except Exception as e:
        return _handle_error("未知错误", str(e))

    # 无问题：正常通过，并写入 commit message 建议
    if not issues:
        print(f"{GREEN}[LLM Review]{NC} {summary.strip()}")
        commit_msg = (result or {}).get("commit_message") or ""
        # 如果 LLM 未返回 commit_message，视为结果不完整，显式报错而不是静默降级
        if not commit_msg.strip():
            raise NetworkError("LLM 未返回 commit_message，无法提供提交信息建议")
        # 仅将建议写入 .git 目录，供 commit-msg 在格式错误时统一提示；
        # 不在 pre-commit 阶段直接打印，避免在正常提交路径制造额外噪音。
        _write_commit_suggestion(commit_msg)
        return 0

    # 将问题按确定性拆分：certain → 硬阻断；uncertain → 交互确认
    certain_issues = []
    uncertain_issues = []
    for issue in issues:
        certainty = (issue.get("certainty") or "").strip().lower()
        if certainty == "certain":
            certain_issues.append(issue)
        else:
            # 任何非 explicit "certain" 的情况一律视为不确定，走交互路径
            uncertain_issues.append(issue)

    print(f"{YELLOW}[LLM Review]{NC} {summary.strip()}")
    print()

    total = len(issues)
    print(
        f"发现 {total} 个潜在问题：{len(certain_issues)} 个确定问题，{len(uncertain_issues)} 个不确定问题"
    )
    print()

    def _print_issue_list(title: str, items: list) -> None:
        if not items:
            return
        print(title)
        for issue in items:
            file = issue.get("file", "unknown")
            line = issue.get("line", "?")
            message = issue.get("message", "未知问题")
            certainty = (issue.get("certainty") or "uncertain").strip().lower()
            certainty_label = "确定" if certainty == "certain" else "不确定"
            # 使用简单的文本标记而非 emoji，避免编码问题
            print(f"[{certainty_label}] {file}:{line}")
            print(f"    {message}")
            print()

    # 先打印确定问题，再打印不确定问题，方便用户区分
    _print_issue_list("以下为确定问题（将直接阻断提交）：", certain_issues)
    _print_issue_list("以下为不确定问题（需要你确认是否继续提交）：", uncertain_issues)

    # 记录本次统计信息
    _record_stats(len(certain_issues), len(uncertain_issues), total)

    # 显示统计摘要（仅在发现问题时显示，帮助观察 LLM 审查趋势）
    _print_stats_summary()

    # 1. 存在确定问题：直接阻断，不允许通过（符合"红线"原则）
    if certain_issues:
        print(f"{RED}[LLM Review]{NC} 检测到确定违反设计原则的问题，提交已被阻断")
        print(
            f"{YELLOW}提示：{NC}请根据上述问题修改代码后重新提交；如需在特殊场景下临时跳过审查，可使用环境变量 COMMIT_HOOKS_NO_REVIEW=1"
        )
        return 1

    # 2. 只有不确定问题：进入交互确认，由用户决定是否继续提交
    if not uncertain_issues:
        # 理论上不会走到这里（前面已处理无问题场景），但为了稳妥做一次兜底
        print(f"{GREEN}[LLM Review]{NC} 未检测到需要阻断的确定问题")
        return 0

    # 等待用户确认（默认 N）
    if prompt_user_continue():
        # 用户选择忽略不确定问题，更新统计
        _update_last_stats_ignored()
        print(f"{GREEN}[LLM Review]{NC} 用户在存在不确定问题的情况下，确认继续提交")
        return 0
    else:
        print(f"{RED}[LLM Review]{NC} 用户在存在不确定问题的情况下取消提交")
        return 1


if __name__ == "__main__":
    sys.exit(main())
