# Image Preprocessing

A beginner-friendly Streamlit app for exploring common image preprocessing
techniques. Upload an image, choose one operation in the sidebar, adjust its
settings, compare the result with the original, and download the result as PNG.

## Features

- Upload JPG, JPEG, PNG, WEBP, and BMP images.
- View the filename, dimensions, channel count, and detected image format.
- Apply grayscale, resize, reshape, rotate, flip, crop, blur, sharpen,
  brightness, contrast, threshold, edge detection, and noise removal.
- Compare original and processed images side by side.
- Download the selected result as a PNG file.
- Reset the operation selector to the original image.

All image processing is performed locally. The app does not need an API key,
database, cloud service, or external service.

## Requirements

- Python 3.10 or later
- Visual Studio Code with the Python extension (recommended)

## Run in VS Code on Windows

Open the project folder in VS Code, open **Terminal → New Terminal**, and run
these commands from the folder containing `app.py`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The final command starts the application in your browser. If PowerShell blocks
virtual-environment activation, use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
image_preprocessing/
├── app.py
├── requirements.txt
├── README.md
└── utils/
    ├── __init__.py
    └── image_processing.py
```

`app.py` contains the Streamlit interface and operation controls.
`utils/image_processing.py` contains image decoding, processing, and PNG export
functions.

## Learning notes

- **Resize** resamples an image to new dimensions and can change the number of
  pixels.
- **Reshape** only rearranges the existing NumPy pixel data. Its requested width
  and height must have the same pixel count as the original, so it is not a
  substitute for resizing.
- Processing uses an RGB representation. The app reports the number of
  channels in the uploaded source image; conversions to RGB are performed
  internally so that the OpenCV operations have consistent input.
