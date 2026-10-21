from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8000"


def get(path: str):
    with urlopen(BASE + path, timeout=10) as response:
        assert response.status == 200
        return json.load(response)


def post(path: str, payload: dict):
    request = Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        assert response.status == 200
        return json.load(response)


catalogue = get("/api/models")
assert len(catalogue) == 6

g1, g2, g3, g4, g5, g6 = catalogue
tests = {
    "g1": post("/api/g1/predict", {"values": g1["example"]}),
    "g2": post("/api/g2/predict", {"values": g2["examples"][next(iter(g2["examples"]))]}),
    "g3": post("/api/g3/predict", {"values": g3["example"]}),
    "g4": post("/api/g4/predict", {"values": {field["name"]: field["default"] for field in g4["fields"]}}),
    "g5": post("/api/g5/predict", {"values": g5["example"]}),
    "g6": post("/api/g6/predict", {"text": g6["example"]}),
}

for key, value in tests.items():
    assert value
    print(key, "OK")

print("Toutes les routes HTTP sont opérationnelles.")
