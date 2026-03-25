#!/usr/bin/env python3
"""
Swift vvs 앱 subprocess 진입점.
stdin:  {"code": "...", "language": "python", "seed": 42}
stdout: {"events": [...], "stats": {...}}
        {"error": "..."} on failure
"""
import sys
import json
import traceback
from pathlib import Path


def main():
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
    except Exception as e:
        err = {"error": f"Invalid JSON input: {e}"}
        sys.stdout.write(json.dumps(err, ensure_ascii=False))
        sys.stdout.flush()
        sys.exit(1)

    code = request.get("code", "")
    language = request.get("language", "python")
    model_dir = request.get("model_dir", "models")
    config_path = request.get("config_path", "config.yaml")
    seed = request.get("seed", None)

    config = {}
    try:
        config = __import__("yaml").safe_load(Path(config_path).read_text()) or {}
    except Exception:
        pass  # config.yaml 없어도 기본값으로 동작

    try:
        from core.pipeline import TypingPipeline
        pipeline = TypingPipeline(config, model_dir=model_dir)
        result = pipeline.generate_timing_plan(code, language, seed=seed)
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        err = {"error": str(e)}
        sys.stdout.write(json.dumps(err, ensure_ascii=False))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
