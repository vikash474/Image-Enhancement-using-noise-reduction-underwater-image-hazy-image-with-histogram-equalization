import cv2
from skimage.metrics import structural_similarity as ssim

# Convert images to grayscale if needed
def convert_to_grayscale(image):
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Load images
original_image = cv2.imread('png/hazy3.png')
compressed_image = cv2.imread('enhancedimage/psnrimage/hazy1.png')

# Check if images are loaded properly
if original_image is None:
    print("Error loading original image.")
if compressed_image is None:
    print("Error loading compressed image.")

# Convert images to grayscale
original_gray = convert_to_grayscale(original_image)
compressed_gray = convert_to_grayscale(compressed_image)

# Print dimensions of the images
# print(f"Original image dimensions: {original_gray.shape}")
# print(f"Compressed image dimensions: {compressed_gray.shape}")

# Resize images to the same dimensions if they differ
if original_gray.shape != compressed_gray.shape:
    compressed_gray = cv2.resize(compressed_gray, (original_gray.shape[1], original_gray.shape[0]))

# Compute SSIM
ssim_value, _ = ssim(original_gray, compressed_gray, full=True)

print("SSIM value:", ssim_value)
