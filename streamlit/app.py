"""Streamlit interface for the Image Preprocessing learning app."""

from __future__ import annotations

import io

import cv2
import streamlit as st

from utils.image_processing import (
    OPERATION_DESCRIPTIONS,
    OPERATIONS,
    decode_image,
    image_to_png_bytes,
    process_image,
)


st.set_page_config(
    page_title="Image Preprocessing",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a theme-aware visual system to the page, sidebar, metrics, and image cards.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-color, rgba(128, 128, 128, 0.22));
    }

    .hero-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        margin: 0 0 1.8rem;
        padding: 1.8rem 2rem;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.22));
        border-radius: 22px;
        background:
            radial-gradient(ellipse at 88% 20%, rgba(59, 130, 246, 0.16), transparent 42%),
            linear-gradient(115deg, rgba(59, 130, 246, 0.09), rgba(20, 184, 166, 0.04));
    }

    .hero-eyebrow, .step-number {
        color: #3b82f6;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-card h1 {
        margin: 0.35rem 0 0.3rem;
        color: var(--text-color);
        font-size: clamp(2rem, 4vw, 2.8rem);
        letter-spacing: -0.045em;
        line-height: 1.1;
    }

    .hero-card p, .sidebar-intro p, .section-heading p {
        margin: 0;
        color: var(--text-color);
        opacity: 0.72;
        line-height: 1.55;
    }

    .hero-badge {
        flex: 0 0 auto;
        padding: 0.55rem 0.85rem;
        border: 1px solid rgba(59, 130, 246, 0.28);
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .section-heading {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin: 1.15rem 0 0.85rem;
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.8rem;
        height: 1.8rem;
        flex: 0 0 auto;
        border-radius: 50%;
        background: rgba(59, 130, 246, 0.12);
    }

    .section-heading h3 {
        margin: 0 0 0.18rem;
        color: var(--text-color);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .section-heading p {
        font-size: 0.87rem;
    }

    .sidebar-intro {
        margin: 0.35rem 0 1.25rem;
        padding: 1rem;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.22));
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), transparent);
    }

    .sidebar-intro span {
        color: #3b82f6;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.12em;
    }

    .sidebar-intro h2 {
        margin: 0.35rem 0;
        color: var(--text-color);
        font-size: 1.12rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        min-height: 6.2rem;
        padding: 1rem 1.1rem;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.22));
        border-radius: 15px;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.06));
    }

    .image-panel-label {
        margin: 0 0 0.2rem;
        color: var(--text-color);
        font-size: 1rem;
        font-weight: 700;
    }

    .image-panel-caption {
        margin: 0 0 0.9rem;
        color: var(--text-color);
        font-size: 0.8rem;
        opacity: 0.65;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero-card {
            align-items: flex-start;
            flex-direction: column;
            padding: 1.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache decoded uploads and deterministic transformations for responsive reruns.
@st.cache_data(show_spinner=False)
def cached_decode_image(image_bytes: bytes):
    return decode_image(image_bytes)


@st.cache_data(show_spinner=False)
def cached_process_image(image_bytes: bytes, operation: str, parameters: dict):
    image, _ = decode_image(image_bytes)
    return process_image(image, operation, parameters)


def reset_operation() -> None:
    """Return the operation selector to its unmodified-image state."""
    st.session_state["operation"] = "Original"


def operation_parameters(
    operation: str,
    image_width: int,
    image_height: int,
) -> dict:
    """Show controls for the selected operation and return their values."""
    sidebar = st.sidebar
    parameters: dict = {}

    if operation == "Resize":
        parameters["width"] = sidebar.number_input(
            "Output width (pixels)",
            min_value=1,
            max_value=20000,
            value=image_width,
            step=1,
        )
        parameters["height"] = sidebar.number_input(
            "Output height (pixels)",
            min_value=1,
            max_value=20000,
            value=image_height,
            step=1,
        )
        parameters["maintain_aspect"] = sidebar.checkbox(
            "Maintain aspect ratio",
            value=False,
        )
        if parameters["maintain_aspect"]:
            output_height = max(
                1,
                round(image_height * parameters["width"] / image_width),
            )
            sidebar.caption(
                f"Effective size: {parameters['width']} × {output_height} pixels"
            )
    elif operation == "Reshape":
        sidebar.caption(
            "Reshape rearranges the existing pixel values; it does not scale the "
            "image. The new width × height must equal the original pixel count."
        )
        parameters["width"] = sidebar.number_input(
            "New width (pixels)",
            min_value=1,
            max_value=20000,
            value=image_width,
            step=1,
        )
        parameters["height"] = sidebar.number_input(
            "New height (pixels)",
            min_value=1,
            max_value=20000,
            value=image_height,
            step=1,
        )
    elif operation == "Rotate":
        parameters["angle"] = sidebar.selectbox(
            "Rotation",
            options=[90, 180, 270, "Custom angle"],
        )
        if parameters["angle"] == "Custom angle":
            parameters["angle"] = sidebar.number_input(
                "Angle (degrees)",
                min_value=-360.0,
                max_value=360.0,
                value=45.0,
                step=1.0,
            )
    elif operation == "Flip":
        parameters["direction"] = sidebar.selectbox(
            "Flip direction",
            options=["Horizontal", "Vertical", "Both"],
        )
    elif operation == "Crop":
        sidebar.caption("Choose the top-left corner and the crop size in pixels.")
        crop_columns = sidebar.columns(2)
        parameters["x"] = crop_columns[0].number_input(
            "Left (x)",
            min_value=0,
            max_value=max(0, image_width - 1),
            value=0,
            step=1,
        )
        parameters["y"] = crop_columns[1].number_input(
            "Top (y)",
            min_value=0,
            max_value=max(0, image_height - 1),
            value=0,
            step=1,
        )
        parameters["width"] = sidebar.number_input(
            "Crop width",
            min_value=1,
            max_value=image_width,
            value=image_width,
            step=1,
        )
        parameters["height"] = sidebar.number_input(
            "Crop height",
            min_value=1,
            max_value=image_height,
            value=image_height,
            step=1,
        )
    elif operation == "Blur":
        parameters["method"] = sidebar.selectbox(
            "Blur method",
            options=["Gaussian Blur", "Median Blur", "Average Blur"],
        )
        parameters["kernel_size"] = sidebar.slider(
            "Kernel size (odd number)",
            min_value=3,
            max_value=31,
            value=5,
            step=2,
        )
    elif operation == "Brightness":
        parameters["value"] = sidebar.slider(
            "Brightness adjustment",
            min_value=-100,
            max_value=100,
            value=0,
            help="Negative values darken the image; positive values brighten it.",
        )
    elif operation == "Contrast":
        parameters["factor"] = sidebar.slider(
            "Contrast factor",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.05,
        )
    elif operation == "Threshold":
        parameters["value"] = sidebar.slider(
            "Threshold value",
            min_value=0,
            max_value=255,
            value=127,
        )
        parameters["mode"] = sidebar.selectbox(
            "Threshold mode",
            options=["Binary", "Binary inverse"],
        )
    elif operation == "Edge Detection":
        parameters["method"] = sidebar.selectbox(
            "Edge method",
            options=["Canny", "Sobel", "Laplacian"],
        )
        if parameters["method"] == "Canny":
            parameters["lower_threshold"] = sidebar.slider(
                "Lower threshold",
                min_value=0,
                max_value=255,
                value=100,
            )
            parameters["upper_threshold"] = sidebar.slider(
                "Upper threshold",
                min_value=0,
                max_value=255,
                value=200,
            )
        elif parameters["method"] == "Sobel":
            parameters["kernel_size"] = sidebar.select_slider(
                "Sobel kernel size",
                options=[1, 3, 5, 7],
                value=3,
            )
        else:
            parameters["kernel_size"] = sidebar.select_slider(
                "Laplacian kernel size",
                options=[1, 3, 5, 7],
                value=3,
            )
    elif operation == "Noise Removal":
        parameters["method"] = sidebar.selectbox(
            "Denoising method",
            options=["Median", "Bilateral", "Non-local Means"],
        )
        if parameters["method"] == "Median":
            parameters["kernel_size"] = sidebar.slider(
                "Kernel size (odd number)",
                min_value=3,
                max_value=15,
                value=5,
                step=2,
            )
        elif parameters["method"] == "Bilateral":
            parameters["diameter"] = sidebar.slider(
                "Filter diameter",
                min_value=3,
                max_value=15,
                value=9,
                step=2,
            )
            parameters["sigma"] = sidebar.slider(
                "Smoothing strength",
                min_value=10,
                max_value=150,
                value=75,
                step=5,
            )
        else:
            parameters["strength"] = sidebar.slider(
                "Denoising strength",
                min_value=1,
                max_value=20,
                value=10,
            )

    return parameters


# Present the app identity and collect the image before enabling any operations.
st.markdown(
    """
    <div class="hero-card">
        <div>
            <div class="hero-eyebrow">Image lab · preprocessing workspace</div>
            <h1>Image Preprocessing</h1>
            <p>Prepare, explore, and compare image transformations in one place.</p>
        </div>
        <div class="hero-badge">● &nbsp;Private · processed locally</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-heading">
        <span class="step-number">01</span>
        <div>
            <h3>Start with an image</h3>
            <p>Choose a JPG, PNG, WEBP, or BMP file from your device.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Insert Image / Upload Image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP, BMP.",
        label_visibility="collapsed",
    )

# Keep the operation menu available, but prevent processing until an image exists.
if "operation" not in st.session_state:
    st.session_state["operation"] = "Original"

st.sidebar.markdown(
    """
    <div class="sidebar-intro">
        <span>WORKSPACE</span>
        <h2>Transform your image</h2>
        <p>Choose a single operation, then adjust its settings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_operation = st.sidebar.radio(
    "Processing operation",
    options=OPERATIONS,
    key="operation",
    disabled=uploaded_file is None,
)
st.sidebar.button(
    "Reset",
    on_click=reset_operation,
    disabled=uploaded_file is None,
    width="stretch",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="section-heading">
            <span class="step-number">02</span>
            <div>
                <h3>Choose a transformation</h3>
                <p>Your operation controls will become available after upload.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Please upload an image to begin preprocessing.")
    st.caption(
        "Your workflow: upload an image → choose one operation → compare and download."
    )
    st.stop()

# Decode the upload once and show its basic properties before applying a filter.
image_bytes = uploaded_file.getvalue()
try:
    original_image, image_info = cached_decode_image(image_bytes)
except (OSError, ValueError) as error:
    st.error(f"This image could not be opened. Please choose a valid image file. ({error})")
    st.stop()

# Summarize source details in compact cards before the operation workspace.
st.markdown(
    """
    <div class="section-heading">
        <span class="step-number">02</span>
        <div>
            <h3>Image details</h3>
            <p>Source image properties are shown below.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
info_columns = st.columns([1.7, 1, 1, 1])
info_columns[0].metric("Filename", uploaded_file.name)
info_columns[1].metric("Width", f"{image_info['width']} px")
info_columns[2].metric("Height", f"{image_info['height']} px")
info_columns[3].metric("Channels", image_info["channels"])
st.caption(f"Detected format: {image_info['format']} · Processing uses RGB pixels")

# Explain the selected technique and provide only its relevant settings.
st.markdown(
    f"""
    <div class="section-heading">
        <span class="step-number">03</span>
        <div>
            <h3>{selected_operation}</h3>
            <p>{OPERATION_DESCRIPTIONS[selected_operation]}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
parameters = operation_parameters(
    selected_operation,
    image_info["width"],
    image_info["height"],
)

# Run just the selected transformation and report invalid settings without crashing.
try:
    if selected_operation == "Original":
        processed_image = original_image
    else:
        processed_image = cached_process_image(
            image_bytes,
            selected_operation,
            parameters,
        )
except (cv2.error, MemoryError, ValueError) as error:
    st.warning(f"Unable to apply {selected_operation.lower()}: {error}")
    st.stop()

# Compare the source and result side by side, then offer a portable PNG download.
st.markdown(
    """
    <div class="section-heading">
        <span class="step-number">04</span>
        <div>
            <h3>Compare your result</h3>
            <p>Review the original beside the currently selected result.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
original_column, processed_column = st.columns(2)
with original_column:
    with st.container(border=True):
        st.markdown('<p class="image-panel-label">Original image</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="image-panel-caption">{image_info["width"]} × '
            f'{image_info["height"]} px · source</p>',
            unsafe_allow_html=True,
        )
        st.image(original_image, width="stretch")
with processed_column:
    with st.container(border=True):
        result_height, result_width = processed_image.shape[:2]
        result_mode = "grayscale" if processed_image.ndim == 2 else "RGB"
        st.markdown('<p class="image-panel-label">Processed image</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="image-panel-caption">{result_width} × '
            f'{result_height} px · {result_mode}</p>',
            unsafe_allow_html=True,
        )
        st.image(processed_image, width="stretch")

try:
    png_bytes = image_to_png_bytes(processed_image)
except (OSError, ValueError) as error:
    st.error(f"Could not prepare the image for download: {error}")
else:
    st.markdown(
        """
        <div class="section-heading">
            <span class="step-number">05</span>
            <div>
                <h3>Export your result</h3>
                <p>Download the processed image as a PNG file.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download Processed Image",
        data=io.BytesIO(png_bytes),
        file_name="processed_image.png",
        mime="image/png",
        width="stretch",
        type="primary",
    )
