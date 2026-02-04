#!/bin/bash
# 文件名: check_sensitive.sh
# 描述: 敏感文件检测模块
# 创建日期: 2026年01月25日 15:32:00
# 最后更新日期: 2026年01月25日 15:32:00

check_sensitive_files() {
    local found_sensitive=0
    local sensitive_files=()

    # 获取暂存区文件列表
    local staged_files
    staged_files=$(git diff --cached --name-only)

    for file in $staged_files; do
        for pattern in "${SENSITIVE_PATTERNS[@]}"; do
            if [[ "$file" == "$pattern" ]]; then
                sensitive_files+=("$file")
                found_sensitive=1
            fi
        done
    done

    if [[ $found_sensitive -eq 1 ]]; then
        echo -e "${RED}[pre-commit] 错误: 检测到敏感文件${NC}"
        echo ""
        echo "以下文件不应提交到仓库:"
        for f in "${sensitive_files[@]}"; do
            echo "  - $f"
        done
        echo ""
        echo "请将这些文件添加到 .gitignore 或从暂存区移除:"
        echo "  git reset HEAD <file>"
        echo ""
        echo -e "${RED}提示：${NC}修复后重新提交，或使用 git commit --no-verify 跳过所有检查"
        return 1
    fi

    echo -e "${GREEN}[pre-commit]${NC} 敏感文件检测通过"
    return 0
}
