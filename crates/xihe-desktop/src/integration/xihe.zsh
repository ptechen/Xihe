# Xihe shell integration (zsh)：发 OSC 133 标记，供终端切分命令块。
# 由 xihe-desktop 通过 ZDOTDIR 注入；先加载用户自己的 .zshrc。
ZDOTDIR="$__XIHE_USER_ZDOTDIR"
if [[ -f "${ZDOTDIR:-$HOME}/.zshrc" ]]; then
  source "${ZDOTDIR:-$HOME}/.zshrc"
fi

__xihe_precmd() {
  # 必须是第一个 precmd 钩子，$? 才是命令的真实退出码。
  local __xihe_status=$?
  printf '\033]133;D;%s\007' "$__xihe_status"
  printf '\033]133;A\007'
}
__xihe_preexec() {
  printf '\033]133;C\007'
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec __xihe_preexec
precmd_functions=(__xihe_precmd $precmd_functions)

# B 标记放在提示符末尾：提示符画完、用户输入开始。
PS1="${PS1}"$'%{\033]133;B\007%}'