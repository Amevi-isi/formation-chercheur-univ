from __future__ import annotations

from app.model_store import (
    catalogue,
    predict_g1,
    predict_g2,
    predict_g3,
    predict_g4,
    predict_g5,
    predict_g6,
)


models = catalogue()
assert len(models) == 6

g1 = models[0]
result_1 = predict_g1(g1["example"])
assert 0 <= result_1["probability"] <= 1

g2 = models[1]
result_2 = predict_g2(g2["examples"][next(iter(g2["examples"]))])
assert abs(sum(item["value"] for item in result_2["probabilities"]) - 1) < 1e-6

g3 = models[2]
result_3 = predict_g3(g3["example"])
assert 15 < result_3["prediction"] < 50

g4 = models[3]
example_4 = {field["name"]: field["default"] for field in g4["fields"]}
result_4 = predict_g4(example_4)
assert 0 <= result_4["probability"] <= 1

g5 = models[4]
result_5 = predict_g5(g5["example"])
assert 20 < result_5["prediction"] < 100

g6 = models[5]
result_6 = predict_g6(g6["example"])
assert abs(sum(item["value"] for item in result_6["probabilities"]) - 1) < 1e-6

for model in models:
    print(model["id"], model["metrics"])

print("RESULTATS")
print("g1", result_1)
print("g2", result_2["prediction"])
print("g3", round(result_3["prediction"], 2))
print("g4", round(result_4["probability"], 3))
print("g5", round(result_5["prediction"], 2))
print("g6", result_6["prediction"])
print("Tous les tests d’inférence sont réussis.")
