"""Image decoding, preprocessing operations, and PNG export helpers."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


OPERATIONS = [
    "Original",
    "Grayscale",
    "Resize",
    "Reshape",
    "Rotate",
    "Flip",
    "Crop",
    "Blur",
    "Sharpen",
    "Brightness",
    "Contrast",
    "Threshold",
    "Edge Detection",
    "Noise Removal",
]

OPERATION_DESCRIPTIONS = {
    "Original": "Shows the uploaded image without modification.",
    "Grayscale": (
        "Converts a color image into shades of gray and reduces the image to "
        "one intensity channel."
    ),
    "Resize": (
        "Changes the width and height of an image by resampling its pixels. "
        "Unlike reshape, resizing can change the total number of pixels."
    ),
    "Reshape": (
        "Changes the NumPy array dimensions without resampling pixels. The new "
        "width and height must have the same pixel count as the original."
    ),
    "Rotate": "Turns the image by a right angle or a custom number of degrees.",
    "Flip": "Mirrors the image horizontally, vertically, or across both axes.",
    "Crop": "Keeps only a rectangular region of the image.",
    "Blur": (
        "Reduces image noise and fine details by smoothing neighboring pixels."
    ),
    "Sharpen": "Emphasizes local edges and details using an unsharp mask.",
    "Brightness": "Adds or subtracts lightness from the image.",
    "Contrast": "Expands or reduces the difference between light and dark pixels.",
    "Threshold": (
        "Converts the image to grayscale, then separates pixels into two "
        "intensity groups using a threshold."
    ),
    "Edge Detection": (
        "Identifies boundaries and sharp intensity changes in an image."
    ),
    "Noise Removal": (
        "Reduces unwanted grain while trying to preserve useful image details."
    ),
}


def decode_image(image_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode uploaded image data into an RGB array and collect source metadata."""
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image_format = source.format or "Unknown"
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            channels = len(image.getbands())
            rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not decode the uploaded image: {error}") from error

    if width < 1 or height < 1:
        raise ValueError("The uploaded image has invalid dimensions.")

    return rgb_image, {
        "width": width,
        "height": height,
        "channels": channels,
        "format": image_format,
    }


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Ensure processing functions receive a non-empty, 8-bit RGB image."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError("Expected a non-empty RGB image.")
    if array.dtype != np.uint8:
        raise ValueError("Image pixel values must use the 8-bit unsigned format.")
    return array


def _validate_kernel_size(kernel_size: int, minimum: int = 1) -> int:
    """Require a positive odd kernel size for neighborhood-based filters."""
    if kernel_size < minimum or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    return kernel_size


def process_image(
    image: np.ndarray,
    operation: str,
    parameters: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply one selected preprocessing operation to an RGB image."""
    rgb_image = _validate_rgb_image(image)
    settings = parameters or {}
    height, width = rgb_image.shape[:2]

    if operation == "Original":
        return rgb_image.copy()
    if operation == "Grayscale":
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    if operation == "Resize":
        new_width = int(settings["width"])
        new_height = int(settings["height"])
        if settings.get("maintain_aspect"):
            new_height = max(1, round(height * new_width / width))
        if new_width < 1 or new_height < 1:
            raise ValueError("Width and height must be greater than zero.")
        if new_width * new_height > 50_000_000:
            raise ValueError("The requested output is too large (limit: 50 megapixels).")
        interpolation = (
            cv2.INTER_AREA
            if new_width < width or new_height < height
            else cv2.INTER_CUBIC
        )
        return cv2.resize(
            rgb_image,
            (new_width, new_height),
            interpolation=interpolation,
        )
    if operation == "Reshape":
        new_width = int(settings["width"])
        new_height = int(settings["height"])
        if new_width < 1 or new_height < 1:
            raise ValueError("Width and height must be greater than zero.")
        if new_width * new_height != width * height:
            raise ValueError(
                f"Invalid shape: {width} × {height} has {width * height} pixels, "
                f"but {new_width} × {new_height} has {new_width * new_height}."
            )
        return rgb_image.reshape((new_height, new_width, 3))
    if operation == "Rotate":
        angle = float(settings["angle"])
        if angle in (90, 180, 270):
            quarter_turns = {90: 3, 180: 2, 270: 1}[int(angle)]
            return np.ascontiguousarray(np.rot90(rgb_image, k=quarter_turns))

        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        output_width = int((height * sine) + (width * cosine))
        output_height = int((height * cosine) + (width * sine))
        matrix[0, 2] += output_width / 2.0 - center[0]
        matrix[1, 2] += output_height / 2.0 - center[1]
        return cv2.warpAffine(
            rgb_image,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    if operation == "Flip":
        direction = settings["direction"]
        flip_code = {"Horizontal": 1, "Vertical": 0, "Both": -1}.get(direction)
        if flip_code is None:
            raise ValueError("Choose Horizontal, Vertical, or Both.")
        return cv2.flip(rgb_image, flip_code)
    if operation == "Crop":
        x = int(settings["x"])
        y = int(settings["y"])
        crop_width = int(settings["width"])
        crop_height = int(settings["height"])
        if (
            x < 0
            or y < 0
            or crop_width < 1
            or crop_height < 1
            or x + crop_width > width
            or y + crop_height > height
        ):
            raise ValueError("The crop rectangle must fit inside the original image.")
        return rgb_image[y : y + crop_height, x : x + crop_width].copy()
    if operation == "Blur":
        kernel_size = _validate_kernel_size(int(settings["kernel_size"]), minimum=3)
        method = settings["method"]
        if method == "Gaussian Blur":
            return cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)
        if method == "Median Blur":
            return cv2.medianBlur(rgb_image, kernel_size)
        if method == "Average Blur":
            return cv2.blur(rgb_image, (kernel_size, kernel_size))
        raise ValueError("Choose a supported blur method.")
    if operation == "Sharpen":
        softened = cv2.GaussianBlur(rgb_image, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(rgb_image, 1.5, softened, -0.5, 0)
    if operation == "Brightness":
        value = float(settings["value"])
        return np.clip(rgb_image.astype(np.float32) + value, 0, 255).astype(
            np.uint8
        )
    if operation == "Contrast":
        factor = float(settings["factor"])
        if factor < 0:
            raise ValueError("Contrast factor cannot be negative.")
        return cv2.convertScaleAbs(rgb_image, alpha=factor, beta=0)
    if operation == "Threshold":
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        threshold_type = (
            cv2.THRESH_BINARY
            if settings["mode"] == "Binary"
            else cv2.THRESH_BINARY_INV
        )
        _, result = cv2.threshold(
            gray_image,
            int(settings["value"]),
            255,
            threshold_type,
        )
        return result
    if operation == "Edge Detection":
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        method = settings["method"]
        if method == "Canny":
            lower = int(settings["lower_threshold"])
            upper = int(settings["upper_threshold"])
            if lower >= upper:
                raise ValueError("The lower threshold must be less than the upper threshold.")
            return cv2.Canny(gray_image, lower, upper)
        kernel_size = _validate_kernel_size(
            int(settings["kernel_size"]),
            minimum=1,
        )
        if method == "Sobel":
            gradient_x = cv2.Sobel(
                gray_image,
                cv2.CV_32F,
                1,
                0,
                ksize=kernel_size,
            )
            gradient_y = cv2.Sobel(
                gray_image,
                cv2.CV_32F,
                0,
                1,
                ksize=kernel_size,
            )
            magnitude = cv2.magnitude(gradient_x, gradient_y)
            return cv2.normalize(
                magnitude,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)
        if method == "Laplacian":
            laplacian = cv2.Laplacian(
                gray_image,
                cv2.CV_32F,
                ksize=kernel_size,
            )
            return cv2.convertScaleAbs(laplacian)
        raise ValueError("Choose Canny, Sobel, or Laplacian.")
    if operation == "Noise Removal":
        method = settings["method"]
        if method == "Median":
            kernel_size = _validate_kernel_size(
                int(settings["kernel_size"]),
                minimum=3,
            )
            return cv2.medianBlur(rgb_image, kernel_size)
        if method == "Bilateral":
            diameter = int(settings["diameter"])
            sigma = float(settings["sigma"])
            if diameter < 1 or sigma <= 0:
                raise ValueError("Filter diameter and smoothing strength must be positive.")
            return cv2.bilateralFilter(
                rgb_image,
                diameter,
                sigma,
                sigma,
            )
        if method == "Non-local Means":
            strength = float(settings["strength"])
            if strength <= 0:
                raise ValueError("Denoising strength must be greater than zero.")
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(
                bgr_image,
                None,
                strength,
                strength,
                7,
                21,
            )
            return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        raise ValueError("Choose a supported noise removal method.")

    raise ValueError(f"Unsupported operation: {operation}")


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a grayscale or RGB result as PNG bytes for downloading."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError("Cannot export an empty or unsupported image.")
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError("Only grayscale and RGB images can be exported.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
