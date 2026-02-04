#!/bin/bash 
# 文件名: install.sh
# 描述: Git 钩子安装脚本 
# 创建日期: 2026年01月25日 15:32:00
# 最后更新日期: 2026年01月27日 02:14:14

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 非 Git 仓库或非 clone 场景（如 sdist 安装）：静默跳过，便于接入 pip 等自动化流程
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    exit 0
fi
if [[ ! -f "${SCRIPT_DIR}/pre-commit" ]]; then
    exit 0
fi

echo "安装 Git 钩子..."

# 优先使用 core.hooksPath：指向本脚本所在目录（支持 commit-hooks 或 scripts/hooks）
if git config --local core.hooksPath "${SCRIPT_DIR}" 2>/dev/null; then
    # Git 要求 core.hooksPath 下的钩子具备可执行位，否则会忽略并 hint
    chmod +x "${SCRIPT_DIR}/pre-commit" "${SCRIPT_DIR}/commit-msg" 2>/dev/null || true
    echo -e "${GREEN}✓${NC} pre-commit 钩子已安装（core.hooksPath）"
    echo -e "${GREEN}✓${NC} commit-msg 钩子已安装（core.hooksPath）"
else
    # 回退：复制到 .git/hooks（Git < 2.9 或不可写 config 时）
    GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"
    mkdir -p "$GIT_HOOKS_DIR"
    cp -f "${SCRIPT_DIR}/pre-commit" "${GIT_HOOKS_DIR}/pre-commit"
    cp -f "${SCRIPT_DIR}/commit-msg" "${GIT_HOOKS_DIR}/commit-msg"
    chmod +x "${GIT_HOOKS_DIR}/pre-commit" 2>/dev/null || true
    chmod +x "${GIT_HOOKS_DIR}/commit-msg" 2>/dev/null || true
    echo -e "${GREEN}✓${NC} pre-commit 钩子已安装（复制）"
    echo -e "${GREEN}✓${NC} commit-msg 钩子已安装（复制）"
fi

echo ""
echo -e "${GREEN}Git 钩子安装完成!${NC}"

# 检查并安装可选依赖（安装失败不影响钩子安装）
echo ""
echo "检查可选依赖..."

# 检测并使用项目虚拟环境（仅支持 .venv）
PYTHON_CMD="python3"
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [[ -f "${GIT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_CMD="${GIT_ROOT}/.venv/bin/python"
    echo -e "${GREEN}ℹ${NC} 使用项目虚拟环境: .venv"
else
    echo -e "${YELLOW}⚠${NC} 未找到项目虚拟环境 (.venv)，使用系统 Python"
    echo "  提示: 运行 'python3 -m venv .venv' 创建虚拟环境"
fi

# 交互式询问函数（仅在交互式终端时询问）
prompt_install() {
    local tool_name=$1
    # install_cmd 参数已移除（未使用，修复 SC2034）
    
    # 非交互式环境（CI、管道等）或已设置环境变量：跳过询问
    if [[ ! -t 0 ]] || [[ "${AUTOMANUS_INSTALL_DEPS:-}" == "1" ]] || [[ "${AUTOMANUS_INSTALL_DEPS:-}" == "0" ]]; then
        if [[ "${AUTOMANUS_INSTALL_DEPS:-}" == "1" ]]; then
            return 0  # 自动安装
        else
            return 1  # 不安装
        fi
    fi
    
    # 交互式询问（修复 SC2018/SC2019：使用 [:upper:] 和 [:lower:]）
    echo -n "是否安装 ${tool_name}? [Y/n] "
    read -r response
    case "$(echo "$response" | tr '[:upper:]' '[:lower:]')" in
        n|no) return 1 ;;
        *) return 0 ;;  # 默认安装
    esac
}

# 检查 black
if ! ${PYTHON_CMD} -m black --version >/dev/null 2>&1; then
    should_install=0
    if [[ "${AUTOMANUS_INSTALL_DEPS:-}" == "1" ]] || prompt_install "black" "pip install black"; then
        should_install=1
    fi
    
    if [[ $should_install -eq 1 ]]; then
        echo "安装 black..."
        # 临时禁用 set -e，安装失败不退出
        set +e
        # 保存错误输出以便诊断
        error_output=$(${PYTHON_CMD} -m pip install black 2>&1)
        install_result=$?
        set -e
        if [[ $install_result -eq 0 ]]; then
            echo -e "${GREEN}✓${NC} black 安装成功"
        else
            echo -e "${YELLOW}⚠${NC} black 安装失败，将跳过格式化检查"
            echo "  错误信息: ${error_output}"
            echo "  提示: 激活虚拟环境后运行 'pip install black' 安装"
        fi
    else
        echo -e "${YELLOW}ℹ${NC} black 未安装（Python 代码格式化检查将跳过）"
        echo "  提示: 运行 'pip install black' 安装"
    fi
else
    echo -e "${GREEN}✓${NC} black 已安装"
fi

# 检查 shellcheck
if ! command -v shellcheck &> /dev/null; then
    should_install=0
    if [[ "${AUTOMANUS_INSTALL_DEPS:-}" == "1" ]] || prompt_install "shellcheck" "brew install shellcheck"; then
        should_install=1
    fi
    
    if [[ $should_install -eq 1 ]]; then
        echo "安装 shellcheck..."
        # 临时禁用 set -e，安装失败不退出
        set +e
        install_result=1
        # 检测系统类型并安装
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS: 使用 brew
            if command -v brew &> /dev/null; then
                brew install shellcheck >/dev/null 2>&1
                install_result=$?
            else
                echo -e "${YELLOW}⚠${NC} 未找到 brew，无法自动安装 shellcheck"
                echo "  提示: 运行 'brew install shellcheck' 安装"
            fi
        elif [[ "$(uname)" == "Linux" ]]; then
            # Linux: 尝试 apt-get 或 yum
            if command -v apt-get &> /dev/null; then
                sudo apt-get install -y shellcheck >/dev/null 2>&1
                install_result=$?
            elif command -v yum &> /dev/null; then
                sudo yum install -y shellcheck >/dev/null 2>&1
                install_result=$?
            else
                echo -e "${YELLOW}⚠${NC} 未找到包管理器，无法自动安装 shellcheck"
            fi
        else
            echo -e "${YELLOW}⚠${NC} 不支持的系统类型，无法自动安装 shellcheck"
        fi
        set -e
        if [[ $install_result -eq 0 ]]; then
            echo -e "${GREEN}✓${NC} shellcheck 安装成功"
        else
            echo -e "${YELLOW}⚠${NC} shellcheck 安装失败，将跳过 Shell 检查"
            echo "  提示: 手动安装 shellcheck"
        fi
    else
        echo -e "${YELLOW}ℹ${NC} shellcheck 未安装（Shell 脚本检查将跳过）"
        echo "  提示:"
        echo "    macOS: brew install shellcheck"
        echo "    Linux: apt-get install shellcheck 或 yum install shellcheck"
    fi
else
    echo -e "${GREEN}✓${NC} shellcheck 已安装"
fi

echo ""
echo "使用说明:"
echo "  - 钩子将在每次 git commit 时自动执行"
echo "  - 使用 AUTOMANUS_NO_REVIEW=1 git commit 跳过 LLM Review"
echo "  - 使用 git commit --no-verify 跳过所有钩子（紧急情况）"
echo ""
echo "可选依赖安装:"
echo "  - 交互式安装: 直接运行脚本，会询问是否安装"
echo "  - 自动安装: AUTOMANUS_INSTALL_DEPS=1 scripts/hooks/install.sh"
echo "  - 跳过询问: AUTOMANUS_INSTALL_DEPS=0 scripts/hooks/install.sh"
