import os
import re

ROOT_DIR = "/ptmp/rfechner/out/exp13_checkpoints"  # <-- change this

pattern = re.compile(r"^global_step_(\d+)$")
dryrun = False
for current_dir, dirnames, _ in os.walk(ROOT_DIR, topdown=False):
    for dirname in dirnames:
        match = pattern.match(dirname)
        if match:
            step = int(match.group(1))
            if step > 80:
                old_path = os.path.join(current_dir, dirname)
                new_name = f"hidden_step_{step}"
                new_path = os.path.join(current_dir, new_name)

                if not os.path.exists(new_path):
                    if not dryrun:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {old_path} -> {new_path}")
                    else:
                        print(f"Dryrun. Would rename: {old_path} -> {new_path}")
                else:
                    print(f"Skipped (target exists): {new_path}")
