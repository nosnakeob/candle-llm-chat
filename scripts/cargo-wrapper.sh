#!/usr/bin/env bash
# MSVC 环境自动加载：检测 link.exe 是否为 MSVC 版，不是则加载 vcvars64.bat
# 兜底：cmd.exe 不可用时（如 Windows 安全策略限制），直接配置 PATH

set -o pipefail

if [[ "$1" != "cargo" ]]; then
  exit 0
fi

link --version 2>&1 | grep -qi microsoft && exit 0

_set_vs_env_via_cmd() {
  local VS_BAT="C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
  exec cmd.exe //c "\"$VS_BAT\" >nul && $*"
}

_set_vs_env_direct() {
  local MSVC_TOOLS="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808"
  export PATH="$MSVC_TOOLS/bin/Hostx64/x64:$PATH"
  cl.exe --help >/dev/null 2>&1 || {
    echo "cargo-wrapper: 错误 - 无法配置 MSVC 编译环境" >&2
    exit 1
  }
  exec "$@"
}

_set_vs_env_via_cmd "$@" 2>/dev/null

# 降级：cmd.exe 不可用，直接 PATH 添加 cl.exe（INCLUDE/LIB 由 .cargo/config.toml [env] 提供）
echo "cargo-wrapper: cmd.exe 不可用，使用 direct PATH 模式" >&2
_set_vs_env_direct "$@"
