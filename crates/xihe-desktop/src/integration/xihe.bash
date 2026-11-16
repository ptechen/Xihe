# Xihe shell integration (bash)：发 OSC 133 标记，供终端切分命令块。
# 由 xihe-desktop 通过 --rcfile 注入；先加载用户自己的 .bashrc。
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

__xihe_precmd() {
  local __xihe_status=$?
  printf '\033]133;D;%s\007' "$__xihe_status"
  printf '\033]133;A\007'
  __xihe_cmd_started=0
}
__xihe_preexec() {
  [[ $__xihe_cmd_started == 1 ]] && return
  [[ $BASH_COMMAND == __xihe_precmd* ]] && return
  __xihe_cmd_started=1
  printf '\033]133;C\007'
}
PROMPT_COMMAND="__xihe_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
PS1="${PS1}\[\033]133;B\007\]"
trap '__xihe_preexec' DEBUG