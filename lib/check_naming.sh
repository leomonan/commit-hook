#!/bin/bash
# 文件名: check_naming.sh
# 描述: 命名模式检测模块
# 创建日期: 2026年01月25日 15:32:00
# 最后更新日期: 2026年01月25日 15:32:00

check_naming_patterns() {
    local found_forbidden=0
    local violations=()

    # 获取暂存区中新增/修改的文件
    local staged_files
    staged_files=$(git diff --cached --name-only --diff-filter=AM)

    for file in $staged_files; do
        # 只检查代码文件
        if [[ ! "$file" =~ \.(py|ts|tsx|js|jsx)$ ]]; then
            continue
        fi

        # 获取文件的 diff 内容（只检查新增的行）
        local diff_content
        diff_content=$(git diff --cached "$file" | grep "^+" | grep -v "^+++" || true)

        for pattern in "${FORBIDDEN_NAMING_PATTERNS[@]}"; do
            # 检查类名定义
            if echo "$diff_content" | grep -qE "(class|interface|type)\s+${pattern}"; then
                local match
                match=$(echo "$diff_content" | grep -oE "(class|interface|type)\s+${pattern}" | head -1)
                violations+=("$file: $match")
                found_forbidden=1
            fi

            # 检查函数名定义
            if echo "$diff_content" | grep -qE "(def|function|const|let|var)\s+${pattern}"; then
                local match
                match=$(echo "$diff_content" | grep -oE "(def|function|const|let|var)\s+${pattern}" | head -1)
                violations+=("$file: $match")
                found_forbidden=1
            fi
        done
    done

    if [[ $found_forbidden -eq 1 ]]; then
        echo -e "${RED}[pre-commit] 错误: 检测到禁止的命名模式${NC}"
        echo ""
        echo "以下命名违反项目规范:"
        for v in "${violations[@]}"; do
            echo "  - $v"
        done
        echo ""
        echo "禁止的命名模式:"
        echo "  - Enhanced*   (暴露实现细节)"
        echo "  - Advanced*   (暴露实现细节)"
        echo "  - *V2         (版本后缀)"
        echo "  - *New        (临时命名)"
        echo "  - *Old        (临时命名)"
        echo "  - *Temp       (临时命名)"
        echo "  - *Backup     (临时命名)"
        echo ""
        echo "建议: 使用描述功能而非实现细节的命名"
        echo ""
        echo -e "${RED}提示：${NC}修复命名后重新提交，或使用 git commit --no-verify 跳过所有检查"
        return 1
    fi

    echo -e "${GREEN}[pre-commit]${NC} 命名模式检测通过"
    return 0
}
