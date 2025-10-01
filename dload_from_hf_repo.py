from huggingface_hub import snapshot_download
import pathlib

dest = pathlib.Path("/data3/mgaur/mllm_inversion")

local_dir = snapshot_download(
    repo_id="manugaur/misc",
    repo_type="model",
    allow_patterns=["refcocog.zip"],
    local_dir=str(dest),              # put files under ./out
    local_dir_use_symlinks=False      # copy real files instead of symlinks
)
print("Files available under:", local_dir)

