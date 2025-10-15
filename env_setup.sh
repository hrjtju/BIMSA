curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

set UV_DEFAULT_INDEX=https://pypi.mirrors.ustc.edu.cn/simple/
uv sync