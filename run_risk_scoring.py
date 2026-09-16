"""Run Notebook 5's report-fusion logic without requiring Jupyter."""

from __future__ import annotations

import json
from pathlib import Path

from claim_analyzer import compute_claim_risk, generate_claim_summary


ROOT = Path(__file__).resolve().parent
with open(ROOT / "data" / "all_videos_visual_reports.json", encoding="utf-8") as file:
    visual_reports = json.load(file)
with open(ROOT / "data" / "all_videos_audio_reports.json", encoding="utf-8") as file:
    audio_reports = json.load(file)

reports = {}
for video_name in sorted(set(visual_reports) & set(audio_reports)):
    visual, audio = visual_reports[video_name], audio_reports[video_name]
    risk = compute_claim_risk(visual, audio)
    reports[video_name] = {
        "visual_report": visual,
        "audio_report": audio,
        "risk_assessment": risk,
        "claim_summary": generate_claim_summary(visual, audio, risk, use_ollama=True),
    }

output_directory = ROOT / "reports"
output_directory.mkdir(exist_ok=True)
with open(output_directory / "all_claim_risk_reports.json", "w", encoding="utf-8") as file:
    json.dump(reports, file, indent=2)

ordered = sorted(reports, key=lambda name: reports[name]["risk_assessment"]["claim_risk_score"])
conflicts = [name for name in ordered if reports[name]["risk_assessment"]["signal_relationship"] == "conflict"]
examples = {"low_risk": ordered[0], "high_risk": ordered[-1]}
if conflicts:
    examples["conflicting_signals"] = conflicts[0]
with open(output_directory / "demo_claim_examples.json", "w", encoding="utf-8") as file:
    json.dump({label: reports[name] | {"video_name": name} for label, name in examples.items()}, file, indent=2)

print(f"Saved {len(reports)} fused claim reports to {output_directory / 'all_claim_risk_reports.json'}")
for label, name in examples.items():
    assessment = reports[name]["risk_assessment"]
    print(f"{label}: {name} | score={assessment['claim_risk_score']} | {assessment['signal_relationship']}")
