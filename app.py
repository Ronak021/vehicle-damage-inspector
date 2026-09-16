"""Streamlit interface for live vehicle-photo and claim-video inspection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from claim_analyzer import analyze_claim_images, analyze_claim_video


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_damage_best.pt"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


def _json_ready(result: dict) -> dict:
    """Remove in-memory annotated images before exporting a report."""
    return {key: value for key, value in result.items() if key not in {"evidence_frames", "evidence_images"}}


def _show_progress_status(title: str):
    status = st.status(title, expanded=True)
    progress_bar = st.progress(8)
    updates: list[str] = []

    def update(message: str) -> None:
        updates.append(message)
        status.write(f"✓ {message}")
        progress_bar.progress(min(18 + len(updates) * 14, 88))

    return status, progress_bar, update


def _damage_table(damage_areas: list[dict]) -> list[dict]:
    return [
        {
            "Damage": item["class"].title(),
            "Confidence": f"{item['confidence']:.0%}",
            "Location": item["location"],
            "Inspection image": item["source_image"],
        }
        for item in damage_areas
    ]


def _render_damage_assessment(assessment: dict) -> None:
    """Part / Damage / Severity / Recommendation report, shown above the raw
    model findings. "Part" is an estimate derived from damage type + bounding-box
    position (see claim_analyzer.estimate_vehicle_part) — the model itself only
    detects damage type, not vehicle panels.
    """
    st.subheader("Vehicle damage assessment")
    st.markdown(f"**Vehicle:** {assessment['vehicle']}  \n**Overall Severity:** {assessment['overall_severity']}")
    rows = assessment.get("rows", [])
    if rows:
        table = [
            {
                "Part": row["part"],
                "Damage": row["damage"],
                "Severity": row["severity"],
                "Recommendation": row["recommendation"],
            }
            for row in rows
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Part is estimated from damage type and position in the photo/frame, not a verified "
            "panel-level detection. A human assessor should confirm the exact part."
        )
    else:
        st.success("No part-level damage crossed the detection threshold.")


def _show_result(result: dict, inspection_type: str, download_name: str) -> None:
    risk = result["risk_assessment"]
    summary = result["claim_summary"]
    visual = result["visual_report"]

    st.subheader("Analysis complete")
    first, second, third = st.columns(3)
    first.metric("Claim risk score", f"{risk['claim_risk_score']:.1f}/100")
    second.metric("Visual damage", f"{risk['visual_damage_score']:.1f}/100")
    third.metric("Priority", risk["risk_band"].title())
    st.info(f"**Recommendation:** {risk['recommendation']}  \n**Signal relationship:** {risk['signal_relationship_note']}")

    _render_damage_assessment(visual["damage_assessment"])

    if inspection_type == "images":
        if visual.get("unreadable_files"):
            st.warning(
                "These uploaded files could not be read as images and were skipped: "
                + ", ".join(visual["unreadable_files"])
            )
        st.subheader("Detected damage")
        st.success(visual["human_readable_explanation"])
        damage_areas = visual["damage_areas"]
        if damage_areas:
            st.dataframe(_damage_table(damage_areas), use_container_width=True, hide_index=True)
        else:
            st.info("No damage type exceeded the configured confidence threshold.")

        st.subheader("Vehicle damage summary")
        left, middle, right = st.columns(3)
        left.metric("Damage areas", visual["damage_area_count"])
        middle.metric("Inspection images", visual["readable_inspection_images"])
        right.metric("AI confidence", f"{visual['average_detection_confidence']:.1f}%")
        for damage in damage_areas:
            st.markdown(
                f"**{damage['class'].title()}**  \n"
                f"Location: {damage['location']}  \n"
                f"Confidence: {damage['confidence']:.0%}"
            )
    else:
        st.subheader("Visual findings")
        if visual["damage_summary"]:
            st.dataframe(visual["damage_summary"], use_container_width=True, hide_index=True)
        else:
            st.success("No damage type crossed the detection threshold.")

    st.subheader("Claim summary")
    st.markdown("**Inspector summary** _(professional / technical)_")
    st.write(summary["inspector_summary"])
    st.markdown("**Customer summary** _(plain language)_")
    st.write(summary["customer_summary"])
    if summary["key_findings"]:
        st.markdown("**Key findings**")
        for finding in summary["key_findings"]:
            st.write(f"• {finding}")
    if summary["flags"]:
        st.markdown("**Review flags**")
        for flag in summary["flags"]:
            st.warning(flag)
    st.caption(f"Summary source: {summary['summary_source']}. Score formula: {risk['score_formula']}.")

    evidence_key = "evidence_images" if inspection_type == "images" else "evidence_frames"
    evidence = result[evidence_key]
    if evidence:
        heading = "Annotated inspection photos" if inspection_type == "images" else "Detected damage — representative frames"
        st.subheader(heading)
        if inspection_type != "images":
            st.caption("One representative frame is shown per unique damage type found, not every sampled frame.")
        for start in range(0, len(evidence), 3):
            columns = st.columns(3)
            for column, item in zip(columns, evidence[start:start + 3]):
                if inspection_type == "images":
                    caption = item["filename"]
                else:
                    damage_label = ", ".join(damage.title() for damage in item["damage_types"])
                    caption = f"{damage_label} · {item['timestamp_sec']}s"
                column.image(item["image"], caption=caption, use_container_width=True)
    else:
        st.caption("No annotated evidence to display — no damage crossed the detection threshold.")

    st.download_button(
        "Download structured claim report (JSON)",
        data=json.dumps(_json_ready(result), indent=2),
        file_name=download_name,
        mime="application/json",
        use_container_width=True,
    )


st.set_page_config(page_title="Vehicle Damage Inspector", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Damage Inspector")
st.caption("Upload inspection photos or a claim walkaround video. The fine-tuned CarDD YOLO model runs locally in the backend.")

if not MODEL_PATH.is_file():
    st.error(f"The fine-tuned model is missing: `models/{MODEL_PATH.name}`. Place it at that path relative to app.py.")
    st.stop()

with st.sidebar:
    st.header("Analysis settings")
    use_ollama = st.checkbox(
        "Generate LangChain + local Ollama narrative",
        value=True,
        help="Uses PromptTemplate → ChatOllama → validated JSON. Falls back to a deterministic summary only if the local LLM is unavailable.",
    )
    st.caption("Claim score: visual damage score, blended with acoustic context for videos when usable audio is present.")
    st.divider()
    st.caption("Decision-support triage only. A human assessor makes all coverage, liability, payment, and fraud decisions.")

vehicle_label = st.text_input(
    "Vehicle (optional)",
    placeholder="e.g., Hyundai Creta 2023",
    help="Shown on the damage assessment report. Leave blank if unknown.",
)

images_tab, video_tab = st.tabs(["📷 Upload vehicle photos", "🎥 Record / upload video"])

with images_tab:
    st.subheader("Upload vehicle photos")
    st.caption("Upload one photo, or many at once — just like the video tab. No fixed angle labels required.")
    uploaded_photos = st.file_uploader(
        "Drag and drop inspection photos or choose files",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="photo_upload",
    )
    total_photos = len(uploaded_photos) if uploaded_photos else 0
    st.caption(f"{total_photos} inspection image{'s' if total_photos != 1 else ''} selected")
    if uploaded_photos:
        preview_columns = st.columns(min(len(uploaded_photos), 5))
        for column, uploaded_file in zip(preview_columns, uploaded_photos[:5]):
            column.image(uploaded_file, use_container_width=True)

    if st.button("Analyze vehicle photos", type="primary", use_container_width=True, disabled=not total_photos):
        with tempfile.TemporaryDirectory(prefix="claim_photos_") as directory:
            entries = []
            for index, uploaded_file in enumerate(uploaded_photos, start=1):
                safe_name = Path(uploaded_file.name).name or f"photo_{index}"
                stored_name = f"{index}_{safe_name}"
                path = Path(directory) / stored_name
                path.write_bytes(uploaded_file.getbuffer())
                entries.append({"filename": safe_name, "path": str(path)})
            status, progress_bar, update = _show_progress_status("Analyzing vehicle photos…")
            status.write("✓ Inspection images received")
            try:
                result = analyze_claim_images(
                    entries, str(MODEL_PATH), vehicle_label=vehicle_label, use_ollama=use_ollama, progress=update
                )
                status.write("✓ Damage identified")
                progress_bar.progress(100)
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as error:
                status.update(label="Analysis could not complete", state="error")
                st.exception(error)
                st.stop()
        _show_result(result, "images", "vehicle_photo_inspection_report.json")

with video_tab:
    st.subheader("Upload inspection video")
    st.caption("Upload a walkaround video. The AI samples the vehicle, detects damage, and analyzes narration when available.")
    upload = st.file_uploader("Drag and drop an inspection video or choose a file", type=VIDEO_TYPES, key="video_upload")
    if upload:
        st.video(upload)
        if st.button("Analyze inspection video", type="primary", use_container_width=True):
            suffix = Path(upload.name).suffix.lower() or ".mp4"
            with tempfile.TemporaryDirectory(prefix="claim_video_") as directory:
                video_path = Path(directory) / f"claim{suffix}"
                video_path.write_bytes(upload.getbuffer())
                status, progress_bar, update = _show_progress_status("Analyzing vehicle video…")
                status.write("✓ Inspection video received")
                try:
                    result = analyze_claim_video(
                        str(video_path), str(MODEL_PATH), vehicle_label=vehicle_label, use_ollama=use_ollama, progress=update
                    )
                    status.write("✓ Damage identified")
                    progress_bar.progress(100)
                    status.update(label="Analysis complete", state="complete", expanded=False)
                except Exception as error:
                    status.update(label="Analysis could not complete", state="error")
                    st.exception(error)
                    st.stop()
            _show_result(result, "video", f"{Path(upload.name).stem}_claim_report.json")

st.divider()
st.caption("Model classes: dent, scratch, crack, glass shatter, lamp broken, tire flat. Bounding boxes are model evidence; the Part column is an estimated location, confirmed by a human assessor.")
