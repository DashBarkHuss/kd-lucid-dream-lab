We added the folder `gssc/` because `gssc-version-6990abd04c0fc72418500e73fd1bc16242dfc8d0/` was our original folder of scripts. But that folder must be used with gssc commit 6990abd04c0fc72418500e73fd1bc16242dfc8d0. Breaking changes were made to the gssc api after that commit. That is why we have a separate folder for the new api. We use conda environemnt `gssc` for this folder, so make sure you `conda activate gssc` before running the scripts. We've updated the conda environment with the latest gssc commit in conda environment `gssc`. Conda environment `base` still uses the old gssc version.

## Checking the Commit Version of GSSC

When working with GSSC, you may need to verify which commit version is installed, especially since the version number (0.0.9) isn't regularly updated. Here's a practical approach:

### Step 1: Identify Recent Changes in the Repository

First, check what has changed in recent commits:

```bash
# Get the latest 3 commits with messages and dates
curl -s https://api.github.com/repos/jshanna100/gssc/commits | jq '.[0:3] | .[] | {sha: .sha, date: .commit.author.date, message: .commit.message}'

# Example output:
# {
#   "sha": "b0420c509a3238867150515a47730841ab28b282",
#   "date": "2024-12-16T23:04:19Z",
#   "message": "basic documentation for ArrayInfer"
# }
```

Then, examine what files were changed in a specific commit:

```bash
# Replace COMMIT_HASH with the actual hash from above
curl -s https://api.github.com/repos/jshanna100/gssc/commits/COMMIT_HASH | jq '.files[].filename'

# Example output:
# "gssc/infer.py"
# "README.md"
```

Finally, look at the actual code changes:

```bash
curl -s https://api.github.com/repos/jshanna100/gssc/commits/COMMIT_HASH | jq '.files[0].patch'
```

Take note of specific code patterns that were added or modified. For example, in December 2024 updates, the `ArrayInfer` class constructor was changed to use `use_cuda` parameter, and a `@torch.no_grad()` decorator was added.

### Step 2: Check Your Installed Version

Now that you know what to look for, check if these changes are present in your installed version:

```bash
# Find the location of your installed package
python -c "import gssc; print(gssc.__file__)"

# Check for specific code patterns identified in Step 1
grep -A 3 "def __init__" /path/to/your/environment/lib/python3.x/site-packages/gssc/infer.py

# Check for other identified patterns
grep -A 2 "@torch.no_grad()" /path/to/your/environment/lib/python3.x/site-packages/gssc/infer.py
```

### Examples of Version-Specific Code

Here are key code signatures for different versions:

**Latest Version (December 2024)**:

- ArrayInfer constructor: `def __init__(self, net=None, con_net=None, use_cuda=False, gpu_idx=None):`
- Uses `@torch.no_grad()` decorator for the `infer` method

**Older Version**:

- ArrayInfer constructor: `def __init__(self, net, con_net, sig_combs, perm_matrix, all_chans, sig_len=2560):`
- No `@torch.no_grad()` decorator

### Updating to the Latest Version

If your version is outdated:

```bash
pip cache purge
pip uninstall -y gssc
pip install git+https://github.com/jshanna100/gssc.git --force-reinstall
```

Then verify the installation by checking for the latest code patterns identified in Step 1.

## CSV Export and Validation

The system provides robust CSV export capabilities through the `CSVManager` class in the `realtime_with_restart` package which helps manage brainflow data.

### Features

- Exact format preservation for compatibility
- Comprehensive data validation
- Sleep stage data integration
- Detailed error reporting

### Usage

```python
from gssc_local.realtime_with_restart.export import CSVManager

# Initialize
csv_manager = CSVManager(board_shim)

# Save data
csv_manager.save_new_data(new_data, is_initial=True)
csv_manager.save_to_csv("output.csv")

# Add sleep stage data
csv_manager.add_sleep_stage_to_csv(sleep_stage=1.0,
                                 next_buffer_id=2.0,
                                 epoch_end_idx=100)

# Validate
csv_manager.validate_saved_csv_matches_original_source("original.csv")
```
