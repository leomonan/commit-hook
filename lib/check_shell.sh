#!/bin/bash
# 文件名: check_shell.sh
# 描述: Shell 脚本检查（shellcheck）
# 创建日期: 2026年01月26日 14:42:11
# 最后更新日期: 2026年02月07日 02:22:36

check_shell_scripts() {
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'

    # 检查 shellcheck 是否安装
    if ! command -v shellcheck &> /dev/null; then
        echo -e "${YELLOW}[pre-commit]${NC} shellcheck 未安装，跳过 Shell 脚本检查"
        echo -e "${YELLOW}提示：${NC}安装 shellcheck 以启用检查:"
        echo "  macOS: brew install shellcheck"
        echo "  Linux: apt-get install shellcheck 或 yum install shellcheck"
        return 0
    fi

    # 获取暂存区中的 Shell 脚本
    staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(sh|bash)$' || true)

    if [[ -z "$staged_files" ]]; then
        echo -e "${GREEN}[pre-commit]${NC} 无修改的 Shell 脚本，跳过检查"
        return 0
    fi

found_errors=0
error_files=()
shellcheck_output=""

# 收集 shellcheck 输出
for file in $staged_files; do
    if ! output=$(shellcheck "$file" 2>&1); then
        error_files+=("$file")
        found_errors=1
        shellcheck_output+="$output"$'\n'
    fi
done

if [[ $found_errors -eq 1 ]]; then
    echo -e "${RED}[pre-commit] 错误: Shell 脚本检查失败${NC}"
    echo ""
    echo "有问题的文件:"
    for file in "${error_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "shellcheck 输出:"
    echo "$shellcheck_output"
    echo ""
    
    # 尝试检测常见问题并提供修复建议
    auto_fixable=0
    fix_suggestions=""
    
    # 检测 SC2018/SC2019: tr 'A-Z' 'a-z' 问题
    if echo "$shellcheck_output" | grep -q "SC2018\|SC2019"; then
        auto_fixable=1
        fix_suggestions+="  - SC2018/SC2019: 将 tr 'A-Z' 'a-z' 改为 tr '[:upper:]' '[:lower:]'"$'\n'
    fi
    
    # 检测 SC2034: 未使用的变量
    if echo "$shellcheck_output" | grep -q "SC2034"; then
        fix_suggestions+="  - SC2034: 删除未使用的变量或添加注释 # shellcheck disable=SC2034"$'\n'
    fi
    
    # 检测 SC2086: 未加引号的变量
    if echo "$shellcheck_output" | grep -q "SC2086"; then
        auto_fixable=1
        fix_suggestions+="  - SC2086: 为变量添加引号，如 \"\$var\" 而不是 \$var"$'\n'
    fi
    
    # 检测 SC2046: 未加引号的命令替换
    if echo "$shellcheck_output" | grep -q "SC2046"; then
        auto_fixable=1
        fix_suggestions+="  - SC2046: 为命令替换添加引号，如 \"\$(cmd)\" 而不是 \$(cmd)"$'\n'
    fi
    
    if [[ -n "$fix_suggestions" ]]; then
        echo -e "${YELLOW}修复建议：${NC}"
        echo "$fix_suggestions"
        echo ""
    fi
    
    # 检查是否可以使用 shfmt 进行格式化（如果可用）
    if command -v shfmt &> /dev/null; then
        echo -e "${YELLOW}提示：${NC}可以使用 shfmt 格式化脚本（仅处理格式，不修复逻辑问题）:"
        for file in "${error_files[@]}"; do
            echo "  shfmt -w $file"
        done
        echo ""
    fi
    
    if [[ $auto_fixable -eq 1 ]]; then
        echo -e "${YELLOW}提示：${NC}部分问题可以自动修复，请根据上述建议手动修复后重新提交"
    else
        echo -e "${YELLOW}提示：${NC}根据 shellcheck 输出和上述建议修复问题后重新提交"
    fi

    echo ""
    
    # 获取脚本所在目录（兼容独立项目 commit-hooks 或仓库内 scripts/hooks）
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local FIX_TOOL="${SCRIPT_DIR}/llm_fix_shellcheck.py"
    
    if [[ -f "$FIX_TOOL" ]]; then
        # 尝试交互式询问
        # 在 git commit 时，stdin 可能被重定向，但我们可以从 /dev/tty 读取
        # 检查 stdout 或 stderr 是否连接到终端（更可靠的检测方式）
        if [[ -t 1 ]] || [[ -t 2 ]]; then
            echo -e "${YELLOW}可选：${NC}检测到 LLM 修复工具，是否尝试自动修复本次 shellcheck 报错？(仅修复报错，不改其他逻辑)"
            echo -n "运行 LLM 修复? [y/N] "
            # 尝试从 /dev/tty 读取（即使 stdin 被重定向也能工作）
            if read -r response < /dev/tty 2>/dev/null; then
                case "$(echo "$response" | tr '[:upper:]' '[:lower:]')" in
                    y|yes)
                        echo -e "${GREEN}[pre-commit]${NC} 正在运行 LLM 修复工具..."
                        local PY_CMD="${PYTHON_CMD:-python3}"
                        if "$PY_CMD" "$FIX_TOOL" --files "${error_files[@]}"; then
                            echo -e "${GREEN}[pre-commit]${NC} 修复成功！重新检查 shellcheck..."
                            # 修复后重新检查 shellcheck，确保修复成功
                            local still_has_errors=0
                            for file in "${error_files[@]}"; do
                                if ! shellcheck "$file" > /dev/null 2>&1; then
                                    still_has_errors=1
                                    break
                                fi
                            done
                            if [[ $still_has_errors -eq 0 ]]; then
                                echo -e "${GREEN}[pre-commit]${NC} shellcheck 检查通过，继续提交流程"
                                # 修复成功且检查通过，返回 0 继续提交
                                return 0
                            else
                                echo -e "${YELLOW}[pre-commit]${NC} 修复后仍有问题，请手动检查"
                                return 1
                            fi
                        else
                            echo -e "${RED}[pre-commit]${NC} LLM 修复失败或未完全修复"
                        fi
                        ;;
                    *)
                        echo "  已跳过自动修复。手动运行命令: ${PYTHON_CMD:-python3} $FIX_TOOL"
                        ;;
                esac
            else
                # 读取失败（非交互式环境），显示提示信息
                echo -e "${YELLOW}提示：${NC}可运行以下命令尝试自动修复:"
                echo "  ${PYTHON_CMD:-python3} $FIX_TOOL"
            fi
        else
            # 非交互式环境，显示提示信息
            echo -e "${YELLOW}提示：${NC}可运行以下命令尝试自动修复:"
            echo "  ${PYTHON_CMD:-python3} $FIX_TOOL"
        fi
    else
        echo "  可在 commit-hooks/lib 目录添加 llm_fix_shellcheck.py 工具辅助修复"
    fi
    echo ""
    
    return 1
fi

echo -e "${GREEN}[pre-commit]${NC} Shell 脚本检查通过"
return 0
}

# 如果直接执行脚本（而非被 source），则调用函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_shell_scripts
    exit $?
fi

# 如果被 source，直接调用函数（pre-commit 钩子会调用）
check_shell_scripts
