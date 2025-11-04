# scripts/promote_or_rollback.py
# Ensure model exists
try:
client.get_registered_model(model_name)
except Exception:
client.create_registered_model(model_name)


if args.mode == "promote":
# find latest run with stage_candidate=true
runs = client.search_runs(
experiment_ids=[client.get_experiment_by_name(cfg["mlflow"]["experiment"]).experiment_id],
filter_string="tags.stage_candidate = 'true'",
order_by=["attributes.start_time DESC"],
max_results=1,
)
if not runs:
raise SystemExit("No candidate run to promote.")
r = runs[0]
ok = (
r.data.metrics.get("recall_at_k", 0) >= thresholds["recall_at_k"] and
r.data.metrics.get("answer_contains_gold_frac", 0) >= thresholds["answer_contains_gold_frac"]
)
if not ok:
raise SystemExit("Candidate did not meet thresholds; not promoting.")


# create a new model version pointing to index paths via tags
mv = client.create_model_version(
name=model_name,
source=r.info.artifact_uri,
run_id=r.info.run_id,
tags={
"index_path": r.data.tags.get("index_path"),
"store_path": r.data.tags.get("store_path"),
"embedder": r.data.params.get("embedder"),
}
)
# move old Production to Archived, set this to Production
for v in client.search_model_versions(f"name='{model_name}'"):
if v.current_stage == "Production":
client.transition_model_version_stage(model_name, v.version, "Archived")
client.transition_model_version_stage(model_name, mv.version, "Production")
print(f"Promoted version {mv.version} to Production")


else: # rollback
versions = sorted(client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version))
prod = [v for v in versions if v.current_stage == "Production"]
if not prod:
raise SystemExit("No Production version to rollback from.")
current = prod[0]
prevs = [v for v in versions if int(v.version) < int(current.version)]
if not prevs:
raise SystemExit("No previous version to rollback to.")
target = prevs[-1]
client.transition_model_version_stage(model_name, current.version, "Archived")
client.transition_model_version_stage(model_name, target.version, "Production")
print(f"Rolled back to version {target.version}")