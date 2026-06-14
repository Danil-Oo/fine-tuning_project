# gpu_monitor.py
import subprocess

import requests
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # разрешаем запросы из Open WebUI


def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        name, util, mem_used, mem_total, temp = result.stdout.strip().split(
            ", "
        )
        return {
            "name": name.strip(),
            "utilization": int(util),
            "mem_used_mb": int(mem_used),
            "mem_total_mb": int(mem_total),
            "temperature": int(temp),
        }
    except Exception as e:
        return {"error": str(e)}


def get_ollama_model():
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=2)
        models = r.json().get("models", [])
        if models:
            m = models[0]
            return {
                "name": m.get("name"),
                "ctx_size": m.get("details", {}).get("context_length", "?"),
                "size_vram": m.get("size_vram", 0) // 1024 // 1024,  # MB
            }
        return {"name": None}
    except:
        return {"name": None}


@app.route("/gpu-stats")
def stats():
    return jsonify({**get_gpu_stats(), "ollama": get_ollama_model()})


if __name__ == "__main__":
    app.run(port=5001, debug=False)
