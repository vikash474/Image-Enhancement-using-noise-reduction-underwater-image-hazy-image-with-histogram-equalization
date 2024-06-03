import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def get_dark_channel(image, patch_size=15):
    """Get the dark channel of the image."""
    min_img = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(flat_dark.size * 0.001)]
    atmospheric_light = np.mean(flat_image[search_idx], axis=0)
    return atmospheric_light

def get_transmission_map(image, atmospheric_light, omega=0.95, patch_size=15):
    """Estimate the transmission map of the image."""
    normed_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(normed_image, patch_size)
    return transmission

def guided_filter(I, p, r, eps):
    """Apply guided filter."""
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    q = mean_a * I + mean_b
    return q

def refine_transmission(image, transmission, r=60, eps=1e-3):
    """Refine the transmission map using guided filtering."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    refined_transmission = guided_filter(gray_image, transmission, r, eps)
    return refined_transmission

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    """Recover the haze-free image."""
    transmission = np.maximum(transmission, t0)
    transmission = cv2.merge([transmission, transmission, transmission])
    recovered_image = (image - atmospheric_light) / transmission + atmospheric_light
    
    # Clip the values to avoid over-brightening and control color saturation
    recovered_image = np.clip(recovered_image, 0, 255)
    return recovered_image.astype(np.uint8)

# Load the hazy image
image = cv2.imread('png/hazi.png')

# Perform dark channel prior dehazing
dark_channel = get_dark_channel(image)
atmospheric_light = estimate_atmospheric_light(image, dark_channel)
transmission = get_transmission_map(image, atmospheric_light)
refined_transmission = refine_transmission(image, transmission)
haze_free_image = recover_image(image, refined_transmission, atmospheric_light)

# Display the haze-free image using Matplotlib

# Convert BGR to RGB for displaying with Matplotlib
haze_free_image_rgb = cv2.cvtColor(haze_free_image, cv2.COLOR_BGR2RGB)

# Create a figure to display the haze-free image
plt.figure(figsize=(6, 6))

# Display the haze-free image
plt.imshow(haze_free_image_rgb)
plt.title('Haze-Free Image')
plt.axis('off')

# Define a function to handle button click event
def download_image(event):
    plt.imsave('haze_free_image.png', haze_free_image_rgb)
    print('Image saved as haze_free_image.png')

# Add a button to download the haze-free image
# download_button = plt.axes([0.7, 0.02, 0.2, 0.075])
# button = Button(download_button, 'Download Image')
# button.on_clicked(download_image)

plt.show()
