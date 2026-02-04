# Commit Hooks — 独立 Git 提交钩子

<!--
文件名: README.md
描述: 独立 Git 提交钩子项目说明与安装配置
创建日期: 2026年02月04日 17:24:37
最后更新日期: 2026年02月04日 17:24:37
-->

## 概述

本目录为**独立可复用的 Git Commit 钩子**，可在任意仓库中使用。提供 pre-commit / commit-msg 检查：敏感文件、命名规范、文件头、格式化、静默异常、可选 LLM Review 等。

## 安装

在**使用该钩子的 Git 仓库根目录**执行（`commit-hooks` 可位于仓库内任意路径或子模块）：

```bash
# 仓库内 commit-hooks 目录
bash commit-hooks/install.sh

# 或指定绝对路径
bash /path/to/commit-hooks/install.sh
```

安装后，`core.hooksPath` 会指向该 `commit-hooks` 目录，钩子常驻此处，拉取更新后无需重复安装。

**可选依赖**（未安装时对应检查会跳过，不阻断提交）：

- **black**：Python 格式化
- **shellcheck**：Shell 脚本检查

自动安装可选依赖：

```bash
AUTOMANUS_INSTALL_DEPS=1 bash commit-hooks/install.sh
```

## 卸载

```bash
bash commit-hooks/uninstall.sh
```

## 配置

### 规则配置（SSOT）

`config.sh` 中可修改：

- `SENSITIVE_PATTERNS` — 敏感文件模式
- `FORBIDDEN_NAMING_PATTERNS` — 禁止命名模式
- `COMMIT_MSG_TYPES` — commit type 白名单
- `PYTHON_HEADER_REQUIRED_FIELDS` — Python 文件头必需字段

### LLM Review 配置

LLM 审查使用独立配置文件，支持：

1. **环境变量**：`COMMIT_HOOKS_LLM_CONFIG` 指定配置文件路径（绝对路径或相对仓库根）。
2. **约定路径**（按顺序查找）：仓库根下 `automanus.llm.toml` 或 `.commit-hooks.llm.toml`。

配置文件需包含 `[global] review_provider = "xxx"` 及对应 `[xxx]` 节（base_url、model、api_key 或 api_key_env）。详见项目内示例或原仓库 `scripts/hooks/README.md`。

### 缓存目录

LLM Review 缓存默认：`~/.cache/commit-hooks/llm_review`。可通过环境变量 `COMMIT_HOOKS_CACHE_DIR` 覆盖。

## 环境变量

| 变量 | 说明 |
|------|------|
| `AUTOMANUS_NO_REVIEW=1` | 跳过 LLM Review |
| `AUTOMANUS_LLM_REVIEW_FORCE=1` | 强制 LLM Review（跳过缓存） |
| `AUTOMANUS_LLM_REVIEW_PRINT_PROMPT=1` | 打印 LLM 提示词（调试） |
| `AUTOMANUS_HOOKS_HELP=1` | 打印帮助并退出，不提交 |
| `AUTOMANUS_INTERACTIVE=1` | 每步前询问是否执行检查 |
| `COMMIT_HOOKS_LLM_CONFIG` | LLM 配置文件路径 |
| `COMMIT_HOOKS_CACHE_DIR` | LLM 缓存目录 |

紧急跳过所有钩子：`git commit --no-verify`。

## 目录结构

```
commit-hooks/
├── README.md
├── install.sh
├── uninstall.sh
├── config.sh
├── pre-commit
├── commit-msg
└── lib/
    ├── check_sensitive.sh
    ├── check_naming.sh
    ├── check_header.py
    ├── check_format.py
    ├── check_shell.sh
    ├── check_no_silent_excepts.py
    └── llm_review.py
```

## 集成到其他项目

- **拷贝**：将本目录复制到仓库内（如 `vendor/commit-hooks` 或根目录 `commit-hooks`），在仓库根执行 `bash commit-hooks/install.sh`。
- **子模块**：`git submodule add <url> commit-hooks`，克隆后执行 `bash commit-hooks/install.sh`。

本钩子不依赖项目名或路径，仅依赖仓库根与 `config.sh` / 可选 LLM 配置。
