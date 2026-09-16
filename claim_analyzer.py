"""Backend for the Insurance Claim Video Analyzer.

The module deliberately keeps the claim score deterministic.  An optional
LangChain/Ollama step only turns the computed evidence into a readable summary;
it never makes, changes, or overrides a score or recommendation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable



CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
SEVERITY_WEIGHTS = {
    "crack": 1.0,
    "glass shatter": 1.0,
    "dent": 0.7,
    "lamp broken": 0.7,
    "tire flat": 0.9,
    "scratch": 0.4,
}
STRESS_FEATURE_WEIGHTS = {
    "onset_rate_per_sec": 0.35,
    "pitch_std_hz": 0.25,
    "rms_std": 0.20,
    "zcr_std": 0.10,
    "pause_ratio": 0.10,
}
# Min/max values calibrated on the project's synthetic narration set.  They are
# intentionally visible rather than hidden inside a learned classifier.
ACOUSTIC_CALIBRATION = {
    "onset_rate_per_sec": (4.99, 6.65),
    "pitch_std_hz": (8.0, 13.1),
    "rms_std": (0.0267, 0.0392),
    "zcr_std": (0.0785, 0.1314),
    "pause_ratio": (0.199, 0.202),
}


def _round(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


_OVERALL_SEVERITY_RANK = {"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3}
_RANK_TO_SEVERITY = {rank: name for name, rank in _OVERALL_SEVERITY_RANK.items()}


def _severity_band(damage_class: str, confidence: float) -> str:
    """Translate a raw detection confidence into a plain-language severity band."""
    weighted = SEVERITY_WEIGHTS.get(damage_class, 0.5) * confidence
    if weighted >= 0.60:
        return "Severe"
    if weighted >= 0.35:
        return "Moderate"
    return "Minor"


def _recommend_action(severity: str) -> str:
    return "Replace" if severity == "Severe" else "Repair"


def estimate_vehicle_part(damage_class: str, bbox_xyxy: list[float], image_width: int, image_height: int) -> str:
    """Best-effort vehicle-panel guess from damage class + bounding-box position.

    The fine-tuned model detects a damage *type* (CarDD classes), not a vehicle
    *part*. There is no panel-segmentation model in this project. This heuristic
    gives assessors a readable "Part" column without pretending to be a verified
    panel-level detector — it should be confirmed by a human assessor, same as
    every other output of this decision-support tool.
    """
    x_center = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    y_center = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    horizontal = "Left" if x_center < image_width / 3 else "Right" if x_center > image_width * 2 / 3 else "Center"
    vertical = "Upper" if y_center < image_height / 3 else "Lower" if y_center > image_height * 2 / 3 else "Middle"

    if damage_class == "lamp broken":
        return "Headlight/Taillight" if horizontal == "Center" else f"{horizontal} Headlight/Taillight"
    if damage_class == "glass shatter":
        return "Windshield / Window Glass"
    if damage_class == "tire flat":
        return "Tire"
    if vertical == "Upper":
        return "Bonnet / Hood" if horizontal == "Center" else f"{horizontal} Fender"
    if vertical == "Lower":
        return "Front / Rear Bumper" if horizontal == "Center" else f"{horizontal} Rocker Panel / Skirt"
    return "Body Panel" if horizontal == "Center" else f"{horizontal} Door Panel"


def build_damage_assessment_table(detections: list[dict[str, Any]], vehicle_label: str = "Not specified") -> dict[str, Any]:
    """Build the Part / Damage / Severity / Recommendation table shown to users.

    Each item in `detections` needs: class, confidence, bbox_xyxy, image_width,
    image_height. Rows are de-duplicated on (part, damage class) — keeping only
    the strongest confidence — so repeated detections of the same damage across
    multiple photos or frames do not show up as separate rows.
    """
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for detection in detections:
        part = estimate_vehicle_part(
            detection["class"], detection["bbox_xyxy"], detection["image_width"], detection["image_height"]
        )
        severity = _severity_band(detection["class"], detection["confidence"])
        key = (part, detection["class"])
        candidate = {
            "part": part,
            "damage": detection["class"].title(),
            "severity": severity,
            "recommendation": _recommend_action(severity),
            "confidence": detection["confidence"],
            "source": detection.get("source"),
        }
        existing = best_rows.get(key)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            best_rows[key] = candidate

    rows = sorted(
        best_rows.values(),
        key=lambda row: (_OVERALL_SEVERITY_RANK[row["severity"]], row["confidence"]),
        reverse=True,
    )
    overall_rank = max((_OVERALL_SEVERITY_RANK[row["severity"]] for row in rows), default=0)
    return {
        "vehicle": (vehicle_label or "Not specified").strip() or "Not specified",
        "overall_severity": _RANK_TO_SEVERITY[overall_rank],
        "rows": rows,
    }


@lru_cache(maxsize=2)
def load_damage_model(model_path: str) -> Any:
    """Load the fine-tuned CarDD YOLO model once per process."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuned model was not found: {path}")
    return YOLO(str(path))


def extract_frames_scene_change(
    video_path: str, diff_threshold: float = 25.0, min_gap_frames: int = 5, max_frames: int = 80
) -> list[dict[str, Any]]:
    """Sample the first frame and meaningful scene changes from a video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    frames: list[dict[str, Any]] = []
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    previous_gray: Any | None = None
    last_capture = -min_gap_frames

    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        changed = previous_gray is None or float(cv2.absdiff(gray, previous_gray).mean()) > diff_threshold
        if changed and frame_index - last_capture >= min_gap_frames:
            frame_height, frame_width = frame.shape[:2]
            frames.append({
                "frame": frame,
                "frame_idx": frame_index,
                "timestamp_sec": _round(frame_index / fps, 2),
                "frame_width": frame_width,
                "frame_height": frame_height,
            })
            previous_gray = gray
            last_capture = frame_index
        frame_index += 1

    cap.release()
    if not frames:
        raise ValueError("No readable frames were found in the uploaded video.")
    return frames


def _annotate_detection_frame(result: Any) -> Any:
    """Return RGB evidence image suitable for Streamlit."""
    import cv2

    return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)


def run_visual_analysis(
    video_path: str,
    model_path: str,
    vehicle_label: str = "Not specified",
    confidence_threshold: float = 0.20,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the project's fine-tuned detector and aggregate frame evidence.

    Evidence frames are de-duplicated: scene-change sampling can pull many
    near-identical frames of the same damage, so instead of returning every
    sampled frame that has a detection, this keeps only the single best
    (highest-confidence) frame per unique damage type actually found.
    """
    frames = extract_frames_scene_change(video_path)
    model = load_damage_model(str(Path(model_path).resolve()))
    frame_results: list[dict[str, Any]] = []
    all_detections: list[dict[str, Any]] = []
    stored_results: list[Any] = []
    best_frame_for_class: dict[str, dict[str, Any]] = {}

    for index, frame_data in enumerate(frames, start=1):
        if progress:
            progress(f"Inspecting frame {index} of {len(frames)} with the fine-tuned damage model…")
        result = model.predict(frame_data["frame"], conf=confidence_threshold, verbose=False)[0]
        stored_results.append(result)
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            confidence = _round(box.conf.item(), 3)
            bbox = [_round(value, 1) for value in box.xyxy[0].tolist()]
            detections.append({"class": label, "confidence": confidence, "bbox_xyxy": bbox})
            all_detections.append({
                "class": label,
                "confidence": confidence,
                "bbox_xyxy": bbox,
                "image_width": frame_data["frame_width"],
                "image_height": frame_data["frame_height"],
                "source": f"{frame_data['timestamp_sec']}s",
            })
            current_best = best_frame_for_class.get(label)
            if current_best is None or confidence > current_best["confidence"]:
                best_frame_for_class[label] = {
                    "frame_list_index": index - 1,
                    "confidence": confidence,
                    "timestamp_sec": frame_data["timestamp_sec"],
                }
        frame_results.append({**frame_data, "detections": detections})

    evidence_by_frame_index: dict[int, dict[str, Any]] = {}
    for damage_class, best in best_frame_for_class.items():
        entry = evidence_by_frame_index.setdefault(
            best["frame_list_index"],
            {"timestamp_sec": best["timestamp_sec"], "damage_types": []},
        )
        entry["damage_types"].append(damage_class)
    evidence: list[dict[str, Any]] = []
    for frame_index in sorted(evidence_by_frame_index):
        entry = evidence_by_frame_index[frame_index]
        entry["damage_types"].sort()
        entry["image"] = _annotate_detection_frame(stored_results[frame_index])
        evidence.append(entry)

    report = aggregate_video_damage_report(frame_results, len(frames))
    report["visual_damage_score"] = compute_visual_damage_score(report["damage_summary"])
    report["video_path"] = str(video_path)
    report["sampling_strategy"] = "scene_change"
    report["confidence_threshold"] = confidence_threshold
    report["damage_assessment"] = build_damage_assessment_table(all_detections, vehicle_label)
    return report, evidence


def _describe_bbox_position(bbox_xyxy: list[float], image_width: int, image_height: int) -> str:
    """Describe where a detection falls within an uploaded photo.

    CarDD labels damage type, not physical vehicle panels or camera angle, so
    this reports bounding-box position only. See `estimate_vehicle_part` for
    the (heuristic) best-guess vehicle part.
    """
    x_center = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
    y_center = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
    horizontal = "left-side" if x_center < image_width / 3 else "right-side" if x_center > image_width * 2 / 3 else "centre"
    vertical = "upper" if y_center < image_height / 3 else "lower" if y_center > image_height * 2 / 3 else "middle"
    return f"{vertical} {horizontal} area"


def run_image_analysis(
    image_entries: list[dict[str, str]],
    model_path: str,
    vehicle_label: str = "Not specified",
    confidence_threshold: float = 0.20,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the fine-tuned damage model over one or more uploaded inspection photos.

    `image_entries` items only need `filename` and `path` — there is no fixed
    "Front/Rear/Left/Right" angle requirement; upload one photo or many, the
    same way the video tab accepts a single file.
    """
    import cv2

    if not image_entries:
        raise ValueError("Upload at least one inspection photo.")
    model = load_damage_model(str(Path(model_path).resolve()))
    photo_results: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    detailed_detections: list[dict[str, Any]] = []
    unreadable_files: list[str] = []

    for index, entry in enumerate(image_entries, start=1):
        if progress:
            progress(f"Inspecting photo {index} of {len(image_entries)}: {entry['filename']}…")
        image = cv2.imread(entry["path"])
        if image is None:
            # Corrupted upload, unsupported format, or a 0-byte file — skip it
            # rather than failing the whole batch.
            unreadable_files.append(entry["filename"])
            continue
        image_height, image_width = image.shape[:2]
        result = model.predict(image, conf=confidence_threshold, verbose=False)[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            bbox = [_round(value, 1) for value in box.xyxy[0].tolist()]
            confidence = _round(box.conf.item(), 3)
            detection = {
                "class": label,
                "confidence": confidence,
                "bbox_xyxy": bbox,
                "location": _describe_bbox_position(bbox, image_width, image_height),
                "source_image": entry["filename"],
            }
            detections.append(detection)
            detailed_detections.append({
                **detection,
                "image_width": image_width,
                "image_height": image_height,
                "source": entry["filename"],
            })
        photo_results.append({"frame_idx": index - 1, "timestamp_sec": index - 1, "detections": detections})
        # Only keep annotated evidence for photos that actually show damage, so a
        # large batch of uploads doesn't flood the report with clean photos.
        if detections:
            evidence.append({"filename": entry["filename"], "image": _annotate_detection_frame(result)})

    if not photo_results:
        raise ValueError(
            "None of the uploaded files could be read as images: " + ", ".join(unreadable_files)
        )

    # Keep the strongest result per image/type so overlapping YOLO boxes are not
    # presented as separate damage areas in the human-facing table.
    strongest: dict[tuple[str, str], dict[str, Any]] = {}
    for detection in detailed_detections:
        key = (detection["source_image"], detection["class"])
        if key not in strongest or detection["confidence"] > strongest[key]["confidence"]:
            strongest[key] = detection
    damage_areas = sorted(strongest.values(), key=lambda item: item["confidence"], reverse=True)

    report = aggregate_video_damage_report(photo_results, len(photo_results))
    report["visual_damage_score"] = compute_visual_damage_score(report["damage_summary"])
    report["analysis_type"] = "inspection_images"
    report["total_inspection_images"] = len(image_entries)
    report["readable_inspection_images"] = len(photo_results)
    report["unreadable_files"] = unreadable_files
    report["damage_areas"] = damage_areas
    report["damage_area_count"] = len(damage_areas)
    report["average_detection_confidence"] = _round(
        100 * sum(item["confidence"] for item in damage_areas) / len(damage_areas), 1
    ) if damage_areas else 0.0
    report["damage_assessment"] = build_damage_assessment_table(detailed_detections, vehicle_label)
    if damage_areas:
        report["human_readable_explanation"] = (
            f"{len(damage_areas)} damage area{'s' if len(damage_areas) != 1 else ''} detected across "
            f"{len(photo_results)} inspection photo{'s' if len(photo_results) != 1 else ''}."
        )
    else:
        report["human_readable_explanation"] = "No damage type crossed the configured detection threshold in the uploaded inspection photos."
    return report, evidence


def aggregate_video_damage_report(frame_results: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"frame_count": 0, "max_confidence": 0.0, "best_timestamp_sec": None, "confidences": []}
    )
    for frame in frame_results:
        seen: set[str] = set()
        for detection in frame["detections"]:
            label, confidence = detection["class"], detection["confidence"]
            current = stats[label]
            current["confidences"].append(confidence)
            if label not in seen:
                current["frame_count"] += 1
                seen.add(label)
            if confidence > current["max_confidence"]:
                current["max_confidence"] = confidence
                current["best_timestamp_sec"] = frame["timestamp_sec"]

    summary = [
        {
            "damage_type": label,
            "frames_detected_in": entry["frame_count"],
            "frame_consistency_pct": _round(100 * entry["frame_count"] / total_frames),
            "max_confidence": _round(entry["max_confidence"], 3),
            "avg_confidence": _round(sum(entry["confidences"]) / len(entry["confidences"]), 3),
            "best_timestamp_sec": entry["best_timestamp_sec"],
        }
        for label, entry in stats.items()
    ]
    summary.sort(key=lambda item: item["max_confidence"], reverse=True)
    return {
        "total_sampled_frames": total_frames,
        "frames_with_any_damage": sum(bool(frame["detections"]) for frame in frame_results),
        "damage_types_found": [item["damage_type"] for item in summary],
        "damage_summary": summary,
    }


def compute_visual_damage_score(damage_summary: list[dict[str, Any]]) -> float:
    if not damage_summary:
        return 0.0
    scores = []
    for damage in damage_summary:
        severity = SEVERITY_WEIGHTS.get(damage["damage_type"], 0.5)
        consistency = damage["frame_consistency_pct"] / 100
        scores.append(severity * damage["max_confidence"] * (0.5 + 0.5 * consistency))
    scores.sort(reverse=True)
    return min(_round((scores[0] + 0.3 * sum(scores[1:])) * 100), 100.0)


def _extract_audio(video_path: str, wav_path: str) -> None:
    """Extract mono 22.05 kHz audio through the ffmpeg bundled with MoviePy."""
    try:
        from moviepy import VideoFileClip  # MoviePy >= 2
    except ImportError:
        from moviepy.editor import VideoFileClip  # MoviePy 1.x
    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            raise ValueError("No audio stream is present in this video.")
        clip.audio.write_audiofile(wav_path, fps=22050, logger=None)
    finally:
        clip.close()


def extract_acoustic_features(video_path: str) -> dict[str, float]:
    """Extract the same acoustic features used in Notebook 3 from upload audio."""
    import librosa
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="claim_audio_") as temporary_directory:
        wav_path = str(Path(temporary_directory) / "audio.wav")
        _extract_audio(video_path, wav_path)
        signal, sample_rate = librosa.load(wav_path, sr=22050)

    trimmed, _ = librosa.effects.trim(signal, top_db=30)
    if len(trimmed) < sample_rate * 0.3:
        trimmed = signal
    if len(trimmed) < sample_rate * 0.1:
        raise ValueError("The video's audio is too short to analyze.")
    duration = len(trimmed) / sample_rate
    f0, voiced, _ = librosa.pyin(
        trimmed, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sample_rate
    )
    voiced_f0 = f0[voiced] if voiced is not None else np.array([])
    onset_envelope = librosa.onset.onset_strength(y=trimmed, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sample_rate)
    rms = librosa.feature.rms(y=trimmed)[0]
    zcr = librosa.feature.zero_crossing_rate(trimmed)[0]
    silence_threshold = np.percentile(rms, 20)
    return {
        "duration_sec": _round(duration, 2),
        "pitch_mean_hz": _round(np.nanmean(voiced_f0), 1) if len(voiced_f0) else 0.0,
        "pitch_std_hz": _round(np.nanstd(voiced_f0), 1) if len(voiced_f0) else 0.0,
        "onset_rate_per_sec": _round(len(onsets) / duration, 2),
        "pause_ratio": _round(np.mean(rms < silence_threshold), 3),
        "rms_mean": _round(np.mean(rms), 4),
        "rms_std": _round(np.std(rms), 4),
        "zcr_std": _round(np.std(zcr), 4),
    }


def compute_acoustic_stress_score(features: dict[str, float]) -> float:
    contributions = []
    for feature, weight in STRESS_FEATURE_WEIGHTS.items():
        low, high = ACOUSTIC_CALIBRATION[feature]
        normalized = min(max((features[feature] - low) / (high - low), 0.0), 1.0)
        contributions.append(normalized * weight)
    return _round(sum(contributions) * 100)


def run_audio_analysis(video_path: str) -> dict[str, Any]:
    try:
        features = extract_acoustic_features(video_path)
        return {"audio_available": True, "acoustic_features": features, "acoustic_stress_score": compute_acoustic_stress_score(features)}
    except Exception as error:
        # Audio is useful supporting evidence, not a requirement for a valid claim video.
        return {"audio_available": False, "acoustic_features": None, "acoustic_stress_score": None, "audio_note": str(error)}


def _band(score: float) -> str:
    return "low" if score < 35 else "moderate" if score < 65 else "high"


def identify_signal_relationship(visual_score: float, acoustic_score: float | None) -> tuple[str, str]:
    if acoustic_score is None:
        return "audio_unavailable", "Visual evidence is available; no acoustic signal was used."
    visual_band, audio_band = _band(visual_score), _band(acoustic_score)
    if visual_band == audio_band:
        return "agreement", f"Visual damage and acoustic stress are both {visual_band}."
    if visual_band == "high" and audio_band == "low":
        return "conflict", "High visual damage is paired with low acoustic stress; retain visual evidence and route for normal human assessment."
    if visual_band == "low" and audio_band == "high":
        return "conflict", "Low visual damage is paired with high acoustic stress; offer human support and check for evidence not visible in the video."
    return "mixed", f"Visual damage is {visual_band} while acoustic stress is {audio_band}; a reviewer should consider both as context."


def compute_claim_risk(visual_report: dict[str, Any], audio_report: dict[str, Any]) -> dict[str, Any]:
    """Compute transparent triage priority—not approval, denial, fraud, or liability."""
    visual_score = float(visual_report["visual_damage_score"])
    acoustic_score = audio_report.get("acoustic_stress_score")
    # Vehicle damage is the primary evidence signal; acoustic indicators are
    # supporting context, so their contribution is deliberately capped.
    audio_weight = 0.20 if acoustic_score is not None else 0.0
    visual_weight = 1.0 - audio_weight
    risk_score = _round(visual_weight * visual_score + audio_weight * float(acoustic_score or 0))
    relationship, relationship_note = identify_signal_relationship(visual_score, acoustic_score)
    if risk_score < 35:
        recommendation = "Routine human review"
    elif risk_score < 65:
        recommendation = "Standard human assessor review"
    else:
        recommendation = "Priority human assessor review"
    return {
        "claim_risk_score": risk_score,
        "risk_band": _band(risk_score),
        "recommendation": recommendation,
        "score_formula": f"{visual_weight:.0%} visual damage score + {audio_weight:.0%} acoustic stress score",
        "visual_damage_score": visual_score,
        "acoustic_stress_score": acoustic_score,
        "signal_relationship": relationship,
        "signal_relationship_note": relationship_note,
        "important_limit": "Decision-support triage only. A qualified human must make all coverage, liability, fraud, and payment decisions.",
    }


def _build_two_summaries(damage_assessment: dict[str, Any], risk: dict[str, Any]) -> dict[str, str]:
    """Build a professional Inspector summary and a plain-language Customer summary
    from the same computed Part/Damage/Severity/Recommendation rows shown in the UI.
    """
    rows = damage_assessment.get("rows", [])
    vehicle = damage_assessment.get("vehicle", "Not specified")

    if not rows:
        inspector = (
            f"The submitted images/frames for {vehicle} show no damage exceeding the configured "
            f"detection threshold. {risk['signal_relationship_note']}"
        )
        customer = "No visible damage was detected in the photos or video you provided."
        return {"inspector_summary": inspector, "customer_summary": customer}

    overall_severity = damage_assessment.get("overall_severity", "Moderate")
    affected_parts = sorted({row["part"] for row in rows})
    replace_parts = sorted({row["part"] for row in rows if row["recommendation"] == "Replace"})
    repair_parts = sorted({row["part"] for row in rows if row["recommendation"] == "Repair"})
    per_part_findings = " ".join(
        f"The {row['part'].lower()} shows {row['damage'].lower()} rated {row['severity'].lower()}."
        for row in rows[:5]
    )

    inspector = (
        f"The submitted images/frames indicate {overall_severity.lower()} damage to {vehicle} involving "
        f"{', '.join(affected_parts)}. {per_part_findings}"
        + (f" Replacement is recommended for {', '.join(replace_parts)}." if replace_parts else "")
        + (f" {', '.join(repair_parts)} appear{'s' if len(repair_parts) == 1 else ''} suitable for repair."
           if repair_parts else "")
        + " Further physical inspection is recommended to rule out hidden structural or mechanical damage."
    )
    customer = (
        f"Your vehicle has visible damage mainly affecting the {', '.join(affected_parts).lower()}. "
        + (f"The {', '.join(replace_parts).lower()} may need replacement. " if replace_parts else "")
        + (f"The {', '.join(repair_parts).lower()} may be repairable. " if repair_parts else "")
        + "A physical inspection is recommended to check for damage that cannot be seen in the photos."
    )
    return {"inspector_summary": inspector, "customer_summary": customer}


def _deterministic_summary(visual: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    damages = visual.get("damage_summary", [])
    findings = [
        f"{item['damage_type']} (max confidence {item['max_confidence']:.0%}, visible in {item['frame_consistency_pct']:.0f}% of sampled frames)"
        for item in damages[:4]
    ] or ["No damage class exceeded the configured detection threshold."]
    flags = []
    if risk["signal_relationship"] in {"conflict", "mixed"}:
        flags.append(risk["signal_relationship_note"])
    if risk["acoustic_stress_score"] is None:
        flags.append("Audio could not be analyzed; the score uses visual evidence only.")
    damage_assessment = visual.get("damage_assessment")
    if damage_assessment is None:
        # Legacy saved reports (e.g. reports/all_claim_risk_reports.json) predate the
        # Part/Severity table and only carry aggregated damage_summary, with no bbox
        # to estimate a part from. Build a best-effort table so old data still gets
        # a meaningful summary instead of silently reporting "no damage".
        fallback_rows = []
        for item in damages:
            severity = _severity_band(item["damage_type"], item["max_confidence"])
            fallback_rows.append({
                "part": "Vehicle body (unspecified area)",
                "damage": item["damage_type"].title(),
                "severity": severity,
                "recommendation": _recommend_action(severity),
                "confidence": item["max_confidence"],
                "source": None,
            })
        overall_rank = max((_OVERALL_SEVERITY_RANK[row["severity"]] for row in fallback_rows), default=0)
        damage_assessment = {
            "vehicle": "Not specified",
            "overall_severity": _RANK_TO_SEVERITY[overall_rank],
            "rows": fallback_rows,
        }
    summaries = _build_two_summaries(damage_assessment, risk)
    return {
        "risk_score": risk["claim_risk_score"],
        "key_findings": findings,
        "flags": flags,
        "recommendation": risk["recommendation"],
        "inspector_summary": summaries["inspector_summary"],
        "customer_summary": summaries["customer_summary"],
        "summary_source": "deterministic_template",
    }


def generate_claim_summary(visual: dict[str, Any], audio: dict[str, Any], risk: dict[str, Any], use_ollama: bool = False) -> dict[str, Any]:
    """Generate a structured narrative with LangChain when Ollama is available.

    The deterministic output is a production-safe fallback and is always used if
    LangChain, Ollama, or valid JSON is unavailable.
    """
    fallback = _deterministic_summary(visual, risk)
    if not use_ollama:
        return fallback
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama

        prompt = PromptTemplate.from_template(
            "You are an insurance claim triage assistant. Use only the supplied computed evidence "
            "(including the damage_assessment Part/Damage/Severity/Recommendation rows). "
            "Do not infer fraud, liability, coverage, or payment. Return only valid JSON with keys "
            "risk_score, key_findings, flags, recommendation, inspector_summary, customer_summary. "
            "inspector_summary: professional/technical language for a claims assessor, referencing the "
            "affected parts, damage types, and repair/replace recommendations. "
            "customer_summary: 2-3 short sentences in simple, plain language for the vehicle owner. "
            "Keep both concise.\n\n"
            "Computed evidence:\n{evidence}"
        )
        evidence = json.dumps({"visual": visual, "audio": audio, "risk": risk}, default=str)
        response = (prompt | ChatOllama(model=model_name, temperature=0) | StrOutputParser()).invoke({"evidence": evidence})
        response = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        generated = json.loads(response)
        required = {"risk_score", "key_findings", "flags", "recommendation", "inspector_summary", "customer_summary"}
        if not required.issubset(generated) or float(generated["risk_score"]) != risk["claim_risk_score"]:
            raise ValueError("LLM output was incomplete or changed the deterministic score.")
        generated["summary_source"] = f"langchain_ollama:{model_name}"
        return generated
    except Exception as error:
        fallback["llm_fallback_reason"] = str(error)
        return fallback


def analyze_claim_video(
    video_path: str,
    model_path: str,
    vehicle_label: str = "Not specified",
    use_ollama: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full uploaded-video pipeline and return serializable results plus evidence."""
    if progress:
        progress("Sampling video and detecting vehicle damage…")
    visual_report, evidence = run_visual_analysis(video_path, model_path, vehicle_label=vehicle_label, progress=progress)
    if progress:
        progress("Extracting audio and calculating acoustic features…")
    audio_report = run_audio_analysis(video_path)
    risk = compute_claim_risk(visual_report, audio_report)
    summary = generate_claim_summary(visual_report, audio_report, risk, use_ollama=use_ollama)
    return {
        "visual_report": visual_report,
        "audio_report": audio_report,
        "risk_assessment": risk,
        "claim_summary": summary,
        "evidence_frames": evidence,
    }


def analyze_claim_images(
    image_entries: list[dict[str, str]],
    model_path: str,
    vehicle_label: str = "Not specified",
    use_ollama: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run live image inspection with the same scoring and LangChain layers."""
    visual_report, evidence = run_image_analysis(image_entries, model_path, vehicle_label=vehicle_label, progress=progress)
    audio_report = {
        "audio_available": False,
        "acoustic_features": None,
        "acoustic_stress_score": None,
        "audio_note": "Photo inspections do not include narration audio.",
    }
    risk = compute_claim_risk(visual_report, audio_report)
    summary = generate_claim_summary(visual_report, audio_report, risk, use_ollama=use_ollama)
    return {
        "visual_report": visual_report,
        "audio_report": audio_report,
        "risk_assessment": risk,
        "claim_summary": summary,
        "evidence_images": evidence,
    }
