#!/bin/bash
# 文件名: uninstall.sh
# 描述: Git 钩子卸载脚本
# 创建日期: 2026年01月25日 15:32:00
# 最后更新日期: 2026年01月25日 21:23:26

set -e

# 非 Git 仓库：静默退出
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

# 颜色定义
GREEN='\033[0;32m'
NC='\033[0m'

echo "卸载 Git 钩子..."

# 若通过 core.hooksPath 安装：移除本地配置
git config --local --unset core.hooksPath 2>/dev/null || true

# 若通过复制安装：删除 .git/hooks 中的钩子
if [[ -f "${GIT_HOOKS_DIR}/pre-commit" ]]; then
    rm "${GIT_HOOKS_DIR}/pre-commit"
    echo -e "${GREEN}✓${NC} pre-commit 钩子已卸载"
fi
if [[ -f "${GIT_HOOKS_DIR}/commit-msg" ]]; then
    rm "${GIT_HOOKS_DIR}/commit-msg"
    echo -e "${GREEN}✓${NC} commit-msg 钩子已卸载"
fi

echo ""
echo -e "${GREEN}Git 钩子卸载完成!${NC}"
