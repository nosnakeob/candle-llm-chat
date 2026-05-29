#!/usr/bin/env bash
# 在 Git Bash 下运行 cargo 前自动加载 VS 环境
# 检测标准：MSVC 的 link.exe 能正常识别则跳过

CMD="$*"

if [[ ! "$CMD" =~ ^cargo ]]; then
  exit 0
fi

# 检查当前 link.exe 是不是 MSVC 版
if link --version 2>&1 | grep -qi microsoft 2>/dev/null; then
  exit 0
fi

VS_BAT="C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"

exec cmd.exe //c "\"$VS_BAT\" >nul && $CMD"
