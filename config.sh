#!/bin/bash
# 文件名: config.sh
# 描述: Git 钩子规则配置文件 - 单一事实来源（SSOT）
# 创建日期: 2026年01月25日 15:32:00
# 最后更新日期: 2026年02月11日 12:03:41
# 以下变量由 source 本文件的 pre-commit/commit-msg 使用
# shellcheck disable=SC2034

# ============================================================
# Hook 帮助提示（SSOT：print_hooks_help_hint 在 lib/hooks_help.sh）
# ============================================================
COMMIT_HOOKS_HELP_HINT="COMMIT_HOOKS_HELP=1 git commit 可查看所有 hook 参数用法"

# ============================================================
# 敏感文件模式
# ============================================================
SENSITIVE_PATTERNS=(
    ".env"
    ".env.*"
    "*.pem"
    "*.key"
    "*credentials*"
    "*secret*"
    ".aws/"
    ".ssh/"
)

# ============================================================
# 禁止的命名模式（正则表达式）
# ============================================================
FORBIDDEN_NAMING_PATTERNS=(
    "Enhanced[A-Z][a-zA-Z]*"      # EnhancedXxx
    "Advanced[A-Z][a-zA-Z]*"      # AdvancedXxx
    "[A-Z][a-zA-Z]*V2"            # XxxV2
    "[A-Z][a-zA-Z]*New"           # XxxNew
    "[A-Z][a-zA-Z]*Old"           # XxxOld
    "[A-Z][a-zA-Z]*Temp"          # XxxTemp
    "[A-Z][a-zA-Z]*Backup"        # XxxBackup
)

# ============================================================
# Commit Message 格式
# ============================================================
COMMIT_MSG_TYPES="feat|fix|refactor|docs|test|chore|perf|build|style|revert"
COMMIT_MSG_PATTERN="^(${COMMIT_MSG_TYPES})\(.+\): .+"

# ============================================================
# Python 文件头必需字段
# ============================================================
PYTHON_HEADER_REQUIRED_FIELDS=(
    "文件名"
    "描述"
    "创建日期"
    "最后更新日期"
)

# ============================================================
# LLM Review 配置
# ============================================================
# 注意：LLM 超时配置以 commit-hooks.llm.toml 中对应 provider 的 timeout 为准
# LLM_REVIEW_TIMEOUT=30  # 已废弃，不再使用
# 通过 COMMIT_HOOKS_NO_REVIEW=1 环境变量关闭
