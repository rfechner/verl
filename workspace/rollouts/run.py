import os
import json
import datetime
import argparse
import subprocess
from typing import Dict, List, Any

def collect_export_vars(config: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in config.items():
        # Convert lists/dicts to JSON strings so Hydra can parse them
        if isinstance(v, (list, dict)):
            val = json.dumps(v)
        elif v is None:
            val = ""
        else:
            val = str(v)
        
        out.append(f"{k}={val}")
    return out

def main():

    # TODO
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashinfer", action="store_true", help="Activate flashinfer conda env instead of verl when set")
    parser.add_argument("--data", required=True, type=str, help='Path to the prompt dataset to load.')
    parser.add_argument("--out", default=None, help='Output path for serialization of results.')
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help='Model Path')

    args = parser.parse_args()
    
    if not args.out:
        date = datetime.datetime.now().isoformat()
        path = f"/u/rfechner/out/default/{date}-output.parquet"
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        args.out = path

    entrypoint_script = '/u/rfechner/verl/workspace/rollouts/rollouts_entrypoint.sh'
    config = {
        "conda_env" : "flashinfer" if args.flashinfer else "verl",
        "data_path" : args.data,
        "save_path" : args.out,
        "model_path" : args.model
    }

    export_list = collect_export_vars(config)
    sbatch_cmd = ["sbatch", f"--export={','.join(export_list)}", entrypoint_script]
    
    try:
        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        print("Job submitted:")
        print(result.stdout.strip())
        if result.stderr:
            print("sbatch stderr:")
            print(result.stderr.strip())
    except subprocess.CalledProcessError as e:
        print("sbatch failed:")
        print(e.stdout)
        print(e.stderr)
        raise

if __name__ == "__main__":
    main()
