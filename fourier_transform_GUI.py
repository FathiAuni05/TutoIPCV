import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Sidebar options
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Function to normalize images
def normalize_image(img):
    img = img / np.max(img)
    return (img * 255).astype("uint8")

# Function to apply mask
def apply_mask(input_image, mask):
    _, mask_thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    mask_bool = mask_thresh.astype(bool)
    input_image[mask_bool] = 1
    return input_image

# Apply mask to all images
def apply_mask_all(list_images, list_mask):
    final_result = []
    for i, mask in zip(list_images, list_mask):
        result = apply_mask(i, mask)
        final_result.append(result)
    return final_result

# Perform FFT on RGB channels
def rgb_fft(image):
    fft_images = []
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(np.abs(rgb_fft)))
    return fft_images, fft_images_log

# Inverse Fourier Transform
def inverse_fourier(image):
    final_image = []
    for c in image:
        channel = np.abs(np.fft.ifft2(c))
        final_image.append(channel)
    final_image_assembled = np.dstack(
        [final_image[0].astype("int"), final_image[1].astype("int"), final_image[2].astype("int")]
    )
    return final_image_assembled

# Create a drawable canvas instance
def create_canvas_draw_instance(background_image, key, height, width):
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(background_image),
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        height=height,
        width=width,
        key=key,
    )
    return canvas_result

# Main function
def main():
    st.header("Fourier Transformation with Drawing Tool")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "png", "jpg"])
    if uploaded_file is not None:
        # Load and display the original image
        original = Image.open(uploaded_file)
        img = np.array(original)
        st.image(img, use_column_width=True, caption="Uploaded Image")

        # Perform FFT on the image
        fft_images, fft_images_log = rgb_fft(img)
        channel_names = ["Red", "Green", "Blue"]
        file_names = ["bg_image_r.png", "bg_image_g.png", "bg_image_b.png"]

        # Save the frequency domain images
        for fft_log, name in zip(fft_images_log, file_names):
            normalized = normalize_image(fft_log)
            cv2.imwrite(name, normalized)

        # Create canvases for each channel
        canvas_images = []
        for i, (name, channel) in enumerate(zip(file_names, channel_names)):
            st.text(f"{channel} Channel in Frequency Domain:")
            canvas = create_canvas_draw_instance(name, key=f"canvas_{i}", height=img.shape[0], width=img.shape[1])
            canvas_images.append(canvas)

        # Process after drawing
        if st.button("Get Result"):
            canvas_data = [c.image_data for c in canvas_images]
            mask_names = ["canvas_mask_r.png", "canvas_mask_g.png", "canvas_mask_b.png"]

            # Save the drawn masks
            for data, name in zip(canvas_data, mask_names):
                if data is not None:
                    cv2.imwrite(name, normalize_image(data[:, :, 3]))

            # Apply masks
            masks = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in mask_names]
            result = apply_mask_all(fft_images, masks)

            # Perform inverse Fourier transform
            transformed = inverse_fourier(result)
            transformed_clipped = np.clip(transformed, 0, 255).astype("uint8")

            st.text("Reconstructed Image from Frequency Domain:")
            st.image(transformed_clipped, use_column_width=True)

if __name__ == "__main__":
    main()
