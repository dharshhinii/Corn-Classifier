from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
SHARED_IMAGE_DIR = Path(os.getenv("SHARED_IMAGE_DIR", "/data/images"))

st.set_page_config(
    page_title="Image Classification UI",
    page_icon="🌽",
    layout="wide",
)

st.title("Three-Class Image Classification")
st.caption(f"FastAPI endpoint: {API_URL}")


def get_json(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def show_result(result: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", result["predicted_class"])
    col2.metric("Confidence", f"{result['confidence']:.2%}")
    col3.metric(
        "Inference time",
        f"{result['timing_ms']['inference']:.2f} ms",
    )

    probability_df = pd.DataFrame(
        {
            "Class": list(result["probabilities"].keys()),
            "Probability": list(result["probabilities"].values()),
        }
    ).sort_values("Probability", ascending=False)

    st.subheader("Class probabilities")
    st.bar_chart(probability_df.set_index("Class"))
    st.dataframe(probability_df, use_container_width=True, hide_index=True)

    st.subheader("Timing")
    timing_df = pd.DataFrame(
        {
            "Stage": list(result["timing_ms"].keys()),
            "Milliseconds": list(result["timing_ms"].values()),
        }
    )
    st.dataframe(timing_df, use_container_width=True, hide_index=True)

    with st.expander("Raw API response"):
        st.json(result)


with st.sidebar:
    st.header("Service")
    try:
        health = get_json("/health")
        if health["status"] == "healthy":
            st.success("API healthy")
            st.write(f"Model: `{health['model_name']}`")
            st.write(f"Type: `{health['model_type']}`")
        else:
            st.warning("API is not ready")
    except requests.RequestException as exc:
        health = None
        st.error(f"API unavailable: {exc}")

    if st.button("Refresh"):
        st.rerun()

    st.divider()
    st.subheader("Model details")
    try:
        st.json(get_json("/model"))
    except requests.RequestException as exc:
        st.warning(f"Unable to read model details: {exc}")


upload_tab, path_tab = st.tabs(["Upload image", "Server image path"])

with upload_tab:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width=420)

        if st.button("Run prediction", type="primary"):
            if health is None:
                st.error("FastAPI is unavailable.")
            else:
                SHARED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                destination = SHARED_IMAGE_DIR / Path(uploaded.name).name
                destination.write_bytes(uploaded.getbuffer())

                try:
                    with st.spinner("Running inference..."):
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"image_path": str(destination)},
                            timeout=60,
                        )
                    if response.status_code == 200:
                        show_result(response.json())
                    else:
                        st.error(
                            f"Prediction failed: {response.status_code} "
                            f"{response.text}"
                        )
                except requests.RequestException as exc:
                    st.error(f"API request failed: {exc}")

with path_tab:
    image_path = st.text_input(
        "Path visible to FastAPI",
        value="/data/images/test.jpg",
    )
    st.info(
        "This path must exist inside the FastAPI container or API server."
    )

    if st.button("Predict from path", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"image_path": image_path},
                timeout=60,
            )
            if response.status_code == 200:
                show_result(response.json())
            else:
                st.error(f"{response.status_code}: {response.text}")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
