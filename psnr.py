import cv2
import numpy as np

def PSNR(original, compressed):
    # Resize images to the same dimensions
    height = min(original.shape[0], compressed.shape[0])
    width = min(original.shape[1], compressed.shape[1])
    original = cv2.resize(original, (width, height))
    compressed = cv2.resize(compressed, (width, height))

    # Convert images to float32 for accurate calculations
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)

    # Calculate mean squared error (MSE)
    mse = np.mean((original - compressed) ** 2)

    if mse == 0:
        return float('inf')

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
# Load the original and enhanced images
original_image = cv2.imread('png/water.png')
enhanced_image = cv2.imread('enhancedimage/psnrimage/water.png')

# Calculate PSNR values separately for original and enhanced images
psnr_original = PSNR(original_image, original_image)
psnr_enhanced = PSNR(original_image, enhanced_image)

# Print PSNR values
print("PSNR value of original image:", psnr_original)
print("PSNR value of enhanced image:", psnr_enhanced)
