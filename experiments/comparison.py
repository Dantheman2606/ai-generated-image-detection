import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from skimage.restoration import estimate_sigma
from skimage.color import rgb2gray
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft2, fftshift
import io
import os


# ---------------------------
# Utility: Show side-by-side
# ---------------------------
def save_side_by_side(pdf, img1, img2, title1, title2, cmap=None):
    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img1, cmap=cmap)
    ax1.set_title(title1)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img2, cmap=cmap)
    ax2.set_title(title2)
    ax2.axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# ---------------------------
# ELA
# ---------------------------
def compute_ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    temp_buffer = io.BytesIO()
    original.save(temp_buffer, format="JPEG", quality=quality)
    temp_buffer.seek(0)
    compressed = Image.open(temp_buffer)

    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return np.array(ela_image)


# ---------------------------
# Main Comparison
# ---------------------------
def compare_images(real_path, ai_path, output_pdf="comparison_report.pdf"):

    with PdfPages(output_pdf) as pdf:

        real_cv = cv2.imread(real_path)
        ai_cv = cv2.imread(ai_path)

        real_rgb = cv2.cvtColor(real_cv, cv2.COLOR_BGR2RGB)
        ai_rgb = cv2.cvtColor(ai_cv, cv2.COLOR_BGR2RGB)

        real_gray = cv2.cvtColor(real_cv, cv2.COLOR_BGR2GRAY)
        ai_gray = cv2.cvtColor(ai_cv, cv2.COLOR_BGR2GRAY)

        # 1️⃣ Originals
        save_side_by_side(pdf, real_rgb, ai_rgb,
                          "Real - Original",
                          "AI - Original")

        # 2️⃣ Noise Map
        real_blur = cv2.GaussianBlur(real_gray, (5,5), 0)
        ai_blur = cv2.GaussianBlur(ai_gray, (5,5), 0)

        real_noise_map = real_gray - real_blur
        ai_noise_map = ai_gray - ai_blur

        real_noise = estimate_sigma(rgb2gray(real_rgb), channel_axis=None)
        ai_noise = estimate_sigma(rgb2gray(ai_rgb), channel_axis=None)

        save_side_by_side(pdf, real_noise_map, ai_noise_map,
                          f"Real Noise Map (σ={real_noise:.4f})",
                          f"AI Noise Map (σ={ai_noise:.4f})",
                          cmap="gray")

        # 3️⃣ FFT Spectrum
        real_fft = np.log(np.abs(fftshift(fft2(real_gray))) + 1)
        ai_fft = np.log(np.abs(fftshift(fft2(ai_gray))) + 1)

        save_side_by_side(pdf, real_fft, ai_fft,
                          "Real FFT Spectrum",
                          "AI FFT Spectrum",
                          cmap="gray")

        # 4️⃣ Edge Map
        real_edges = cv2.Canny(real_gray, 100, 200)
        ai_edges = cv2.Canny(ai_gray, 100, 200)

        real_sharp = cv2.Laplacian(real_gray, cv2.CV_64F).var()
        ai_sharp = cv2.Laplacian(ai_gray, cv2.CV_64F).var()

        save_side_by_side(pdf, real_edges, ai_edges,
                          f"Real Edges (Sharp={real_sharp:.1f})",
                          f"AI Edges (Sharp={ai_sharp:.1f})",
                          cmap="gray")

        # 5️⃣ ELA
        real_ela = compute_ela(real_path)
        ai_ela = compute_ela(ai_path)

        save_side_by_side(pdf, real_ela, ai_ela,
                          "Real ELA",
                          "AI ELA")

        # 6️⃣ Color Histogram (new page)
        fig = plt.figure(figsize=(10,5))

        ax1 = fig.add_subplot(1,2,1)
        for i, color in enumerate(('b','g','r')):
            hist = cv2.calcHist([real_cv],[i],None,[256],[0,256])
            ax1.plot(hist, color=color)
        ax1.set_title("Real - Color Histogram")

        ax2 = fig.add_subplot(1,2,2)
        for i, color in enumerate(('b','g','r')):
            hist = cv2.calcHist([ai_cv],[i],None,[256],[0,256])
            ax2.plot(hist, color=color)
        ax2.set_title("AI - Color Histogram")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 7️⃣ Numerical Summary Page
        fig = plt.figure(figsize=(8,4))
        plt.axis("off")

        summary_text = (
            "NUMERICAL SUMMARY\n\n"
            f"Noise Sigma → Real: {real_noise:.5f} | AI: {ai_noise:.5f}\n"
            f"Sharpness → Real: {real_sharp:.2f} | AI: {ai_sharp:.2f}\n"
            f"File Size (KB) → Real: {os.path.getsize(real_path)/1024:.1f} | "
            f"AI: {os.path.getsize(ai_path)/1024:.1f}"
        )

        plt.text(0.1, 0.5, summary_text, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nSaved comparison report to: {output_pdf}")

if __name__ == "__main__":
    compare_images("real_image.jpg", "ai_image.jpg")