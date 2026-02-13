#!/bin/bash
# 文件名: hooks_help.sh
# 描述: Hook 帮助提示函数（SSOT，供 pre-commit/commit-msg 共用）
# 创建日期: 2026年02月11日 12:03:41
# 最后更新日期: 2026年02月11日 12:03:41
# 依赖: 需在 source config.sh 且已定义 YELLOW、NC 后 source 本文件

print_hooks_help_hint() {
    echo -e "${YELLOW}提示：${NC}${COMMIT_HOOKS_HELP_HINT}"
}
