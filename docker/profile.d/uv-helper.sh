# Explicitly set the uv cache directory to a persistent location (if not already set).
# This is useful for users who want to make package installation super fast across multiple sessions.
# https://docs.astral.sh/uv/concepts/cache/
# https://docs.astral.sh/uv/reference/environment/#uv_cache_dir
if [[ -d "$HOME" && -n "$HOME" && "$HOME" != "/" && -z "$UV_CACHE_DIR" ]]; then
    uv_cache="$HOME/.cache/uv"
    export UV_CACHE_DIR="$uv_cache"
fi