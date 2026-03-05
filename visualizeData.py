import cv2
import numpy as np

from readData import debayerPolarImage,repackPolarToMosaic

def detect_sobel_edges(image):
    """
    Detect Sobel edges in a 4-channel polarized image and return them in a 3-channel image.

    Parameters:
        image (numpy.ndarray): Input image with 4 channels (polarizations: 0°, 45°, 90°, 135°).

    Returns:
        numpy.ndarray: 3-channel image containing Sobel edge detections.
    """
    # Split the 4-channel image into individual polarization channels
    polar_0, polar_45, polar_90, polar_135 = debayerPolarImage(image)# cv2.split(image)

    # Initialize an empty list to store edge-detected channels
    edges = []

    # Apply Sobel edge detection for each channel
    for channel in [polar_0, polar_45, polar_90, polar_135]:
        # Compute Sobel gradients in x and y directions
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize the magnitude to the range [0, 255] and convert to uint8
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Append the result to the edges list
        edges.append(magnitude_normalized)

    # Combine the first three edge-detected channels into a 3-channel image
    # (Choose polarizations 0°, 45°, 90° for the output channels)
    result = cv2.merge(edges[:4])

    return result



def tenengrad_focus_measure(image, ksize=3):
    """
    Compute the Tenengrad focus measure of an image.
    
    Parameters:
        image (numpy.ndarray): Input image (BGR or grayscale).
        ksize (int): Kernel size for Sobel operator (must be 1, 3, 5, or 7).
    
    Returns:
        float: Tenengrad focus measure value.
    """
    # Convert to grayscale if image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Sobel operator in X and Y directions
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Compute gradient magnitude
    gxy = gx**2 + gy**2
    
    # Return mean gradient magnitude (Tenengrad measure)
    return np.mean(gxy)


def determine_intensity_region(image, threshold=0.1):
    """
    Determines the region of the image with the highest intensity values.

    Parameters:
        image (numpy.ndarray): A 4-channel image (H, W, 4) with intensity values.
        threshold (float): A value between 0 and 1 to decide if intensity changes are significant.

    Returns:
        str: One of "Unknown", "Bottom Left", "Top Left", "Top", "Top Right", "Bottom Right", or "Bottom".
    """
    #if image.shape[-1] != 4:
    #    raise ValueError("Input image must have 4 channels.")

    # Convert image to grayscale by summing up all channels
    gray_image = np.sum(image, axis=-1)

    # Get image dimensions
    height, width = gray_image.shape

    # Define regions
    top_left     = gray_image[:height//2, :width//2]
    top          = gray_image[:height//2, width//4:(3*width)//4]
    top_right    = gray_image[:height//2, width//2:]
    bottom_left  = gray_image[height//2:, :width//2]
    bottom       = gray_image[height//2:, width//4:(3*width)//4]
    bottom_right = gray_image[height//2:, width//2:]

    # Compute average intensities for each region
    regions = {
        "Top Left": np.mean(top_left),
        "Top": np.mean(top),
        "Top Right": np.mean(top_right),
        "Bottom Left": np.mean(bottom_left),
        "Bottom": np.mean(bottom),
        "Bottom Right": np.mean(bottom_right)
    }

    print("determine_intensity_region: ",regions)

    # Find the region with the highest intensity
    max_region = max(regions, key=regions.get)
    max_value  = regions[max_region]

    # Calculate the overall mean intensity
    overall_mean = np.mean(gray_image)

    # If the difference between the highest intensity and the mean is below the threshold, return "Unknown"
    if (max_value - overall_mean) / overall_mean < threshold:
        return "Unknown"

    return max_region



def adjust_contrast(image: np.ndarray, factor: float):
    """
    Adjusts the contrast of an RGB image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array with shape (H, W, 3).
        factor (float): Contrast adjustment factor (1.0 = no change, >1 increases contrast, <1 decreases contrast).
    
    Returns:
        np.ndarray: Contrast-adjusted image.
    """
    # Convert to float for precision
    img_float = image.astype(np.float32) / 255.0
    
    # Compute mean intensity per channel
    mean      = np.mean(img_float, axis=(0, 1), keepdims=True)
    
    # Apply contrast adjustment
    adjusted  = mean + factor * (img_float - mean)
    
    # Clip values to valid range and convert back to uint8
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    
    return adjusted


def convertRGBCVMATToRGB(rgb_image,brightness=0,contrast=0):
    brightnessValue = 10* brightness
    contrastValue   = 1.0 + contrast/10
    rgb_image = adjust_contrast(rgb_image,contrastValue)
    return rgb_image



      
# Normalize each independently to 0-255 for visibility
def normalize_to_u8(img):
              minv = np.min(img)
              maxv = np.max(img)
              if maxv > minv:
                  img = (img - minv) / (maxv - minv)
              else:
                  img = np.zeros_like(img)
              return (img * 255.0).astype(np.uint8)

def convertPolarCVMATToRGB(image,way=0,brightness=0,contrast=0):
    if image is None:
        print("Error: Unable to read the image.")
        return None

    height, width, channels = image.shape
    #if channels == 3: 
    #    print("Casting RGB image as monochrome")
    #    image = image[:,:,0]
    image = image[:,:,0]

    # Split into polarization images
    #from readData import debayerPolarImage
    polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg = debayerPolarImage(image)

    # Create an RGB image
    rgb_image = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)

    brightnessValue = 10* brightness
    contrastValue   = 1.0 + contrast/10

    print("Image Visualization using: ",way)

    # Assign each polarization image to a specific channel
    if (way==0):
      rgb_image[:, :, 0] = np.clip(polarization_0_deg.astype(np.float32)   + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_90_deg.astype(np.float32)  + brightnessValue, 0, 255)
    elif (way==1):
      rgb_image[:, :, 0] = np.clip(polarization_45_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_90_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==2):
      rgb_image[:, :, 0] = np.clip( ( polarization_0_deg.astype(np.float32) +  polarization_45_deg.astype(np.float32) ) / 2   + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip( ( polarization_90_deg.astype(np.float32) + polarization_135_deg.astype(np.float32) ) / 2  + brightnessValue, 0, 255)
    elif (way==4):
      #this needs some care so that values are not clipped
      sumMat = (polarization_0_deg.astype(np.float32) + 
                  polarization_45_deg.astype(np.float32) + 
                  polarization_90_deg.astype(np.float32) + 
                  polarization_135_deg.astype(np.float32)) / 4
      rgb_image[:, :, 0] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==5):
      rgb_image[:, :, 0] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==6):
      rgb_image[:, :, 0] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==7):
      rgb_image[:, :, 0] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==8):
      rgb_image[:, :, 0] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way>=9) and (way<=22):
      # --- Polarization analysis / Stokes-based visualizations ---
      # NOTE:
      #   From a 4-angle DoFP sensor (0/45/90/135), we can compute s0,s1,s2.
      #   s3 (circular component) normally requires additional measurements (e.g., a retarder),
      #   so we set s3 = 0 here unless you later provide R/L (or QWP) channels.

      I0   = polarization_0_deg.astype(np.float32)
      I45  = polarization_45_deg.astype(np.float32)
      I90  = polarization_90_deg.astype(np.float32)
      I135 = polarization_135_deg.astype(np.float32)

      # Linear Stokes (per-pixel)
      S0 = I0 + I90
      S1 = I0 - I90
      S2 = I45 - I135
      S3 = np.zeros_like(S0, dtype=np.float32)

      eps = 1e-6

      # Intensity (average of the 4 angles; stays in 0..255 domain)
      intensity_f = (I0 + I45 + I90 + I135) / 4.0
      intensity_u8 = np.clip(intensity_f + brightnessValue, 0, 255).astype(np.uint8)

      # DoLP (linear) and DoP (total)
      dolp = np.sqrt(S1*S1 + S2*S2) / (S0 + eps)
      dolp = np.clip(dolp, 0.0, 1.0)
      dop  = np.sqrt(S1*S1 + S2*S2 + S3*S3) / (S0 + eps)
      dop  = np.clip(dop, 0.0, 1.0)

      dolp_u8 = (dolp * 255.0).astype(np.uint8)
      dop_u8  = (dop  * 255.0).astype(np.uint8)

      # AoLP in radians [-pi/2, pi/2], map to OpenCV HSV hue [0..179]
      aolp = 0.5 * np.arctan2(S2, S1)
      hue = ((aolp + (np.pi/2.0)) / np.pi) * 179.0
      hue = np.clip(hue, 0.0, 179.0).astype(np.uint8)

      # Ellipticity angle chi in [-pi/4, pi/4]
      # chi = 0.5 * atan2(S3, sqrt(S1^2 + S2^2))
      chi = 0.5 * np.arctan2(S3, np.sqrt(S1*S1 + S2*S2) + eps)

      # DoCP (degree of circular polarization) often defined as |S3|/S0
      docp = np.abs(S3) / (S0 + eps)
      docp = np.clip(docp, 0.0, 1.0)
      docp_u8 = (docp * 255.0).astype(np.uint8)

      # ToP (type of polarization): 0 = linear, 1 = circular (approx from |chi|)
      top = np.abs(4.0 * chi / np.pi)  # chi in [-pi/4, pi/4] -> top in [0,1]
      top = np.clip(top, 0.0, 1.0)
      top_u8 = (top * 255.0).astype(np.uint8)

      # --- Retardation-ish magnitude (assumption-dependent) ---
      # If we (strongly) assume the light is fully polarized (DoP≈1), then:
      #   S0^2 = S1^2 + S2^2 + S3^2  =>  |S3| = sqrt(max(S0^2 - S1^2 - S2^2, 0))
      # This provides ONLY a magnitude for the circular component, not handedness.
      S3_mag = np.sqrt(np.maximum(S0*S0 - S1*S1 - S2*S2, 0.0))
      chi_mag = 0.5 * np.arctan2(S3_mag, np.sqrt(S1*S1 + S2*S2) + eps)  # in [0, pi/4]
      retard_mag = np.clip((4.0 * chi_mag / np.pi), 0.0, 1.0)
      retard_u8 = (retard_mag * 255.0).astype(np.uint8)

      # Helpers for s0..s3 visualization
      s0_u8 = np.clip((S0 * 0.5) + brightnessValue, 0, 255).astype(np.uint8)  # divide by 2 to fit in 8-bit
      # s1,s2,s3 in ~[-255,255] -> map to [0,255] with mid=128
      def _stokes_signed_to_u8(S):
          S = np.clip(S, -255.0, 255.0)
          return (S * 0.5 + 128.0).astype(np.uint8)

      s1_u8 = _stokes_signed_to_u8(S1)
      s2_u8 = _stokes_signed_to_u8(S2)
      s3_u8 = _stokes_signed_to_u8(S3)

      # --- Dispatch ---
      if (way==11):
         # Intensity (grayscale)
         rgb_image[:, :, 0] = intensity_u8
         rgb_image[:, :, 1] = intensity_u8
         rgb_image[:, :, 2] = intensity_u8

      elif (way==10):
         # DoLP (false color)
         rgb_image = cv2.applyColorMap(dolp_u8, cv2.COLORMAP_TURBO)

      elif (way==9):
         # AoLP (HSV): Hue=AoLP, Sat=DoLP, Val=Intensity
         hsv = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)
         hsv[:, :, 0] = hue
         hsv[:, :, 1] = dolp_u8
         hsv[:, :, 2] = intensity_u8
         rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

      elif (way==12):
         # s0 (grayscale)
         rgb_image[:, :, 0] = s0_u8
         rgb_image[:, :, 1] = s0_u8
         rgb_image[:, :, 2] = s0_u8

      elif (way==13):
         # s1
         rgb_image = cv2.applyColorMap(s1_u8, cv2.COLORMAP_TURBO)

      elif (way==14):
         # s2
         rgb_image = cv2.applyColorMap(s2_u8, cv2.COLORMAP_TURBO)

      elif (way==15):
         # s3 (will be ~0 with 4-angle DoFP input)
         rgb_image = cv2.applyColorMap(s3_u8, cv2.COLORMAP_TURBO)

      elif (way==16):
         # AoLP (light): Hue=AoLP, Sat=DoP, Val=255
         hsv = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)
         hsv[:, :, 0] = hue
         hsv[:, :, 1] = dop_u8
         hsv[:, :, 2] = 255
         rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

      elif (way==17):
         # AoLP (dark): Hue=AoLP, Sat=DoP, Val=Intensity
         hsv = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)
         hsv[:, :, 0] = hue
         hsv[:, :, 1] = dop_u8
         hsv[:, :, 2] = intensity_u8
         rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

      elif (way==18):
         # DoP (false color)
         rgb_image = cv2.applyColorMap(dop_u8, cv2.COLORMAP_TURBO)

      elif (way==19):
         # DoCP (false color)  (likely 0 with 4-angle DoFP input)
         rgb_image = cv2.applyColorMap(docp_u8, cv2.COLORMAP_TURBO)

      elif (way==20):
         # ToP (false color): 0=linear, 1=circular (approx; needs S3 for non-zero)
         rgb_image = cv2.applyColorMap(top_u8, cv2.COLORMAP_TURBO)

      elif (way==22):
         # Retardation-ish magnitude (false color; magnitude-only, assumption-dependent)
         rgb_image = cv2.applyColorMap(retard_u8, cv2.COLORMAP_TURBO)
      else:
         # CoP (chirality): Hue encodes chi sign/magnitude, shown at full value
         # Map chi [-pi/4, pi/4] -> hue [0..179]
         hue_chi = ((chi + (np.pi/4.0)) / (np.pi/2.0)) * 179.0
         hue_chi = np.clip(hue_chi, 0.0, 179.0).astype(np.uint8)
         hsv = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)
         hsv[:, :, 0] = hue_chi
         hsv[:, :, 1] = 255
         hsv[:, :, 2] = 255
         rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif (way==24):
         # Surface Normal visualization (heuristic, from AoLP+DoLP)
         # NOTE: This is NOT metric shape-from-polarization.
         # We use AoLP as azimuth (shifted by +90deg for diffuse-reflection convention)
         # and DoLP as a tilt magnitude proxy (mapped to [0, pi/2]).
         I0   = polarization_0_deg.astype(np.float32)
         I45  = polarization_45_deg.astype(np.float32)
         I90  = polarization_90_deg.astype(np.float32)
         I135 = polarization_135_deg.astype(np.float32)

         S0 = I0 + I90
         S1 = I0 - I90
         S2 = I45 - I135
         eps = 1e-6

         dolp = np.sqrt(S1*S1 + S2*S2) / (S0 + eps)
         dolp = np.clip(dolp, 0.0, 1.0)

         aolp = 0.5 * np.arctan2(S2, S1)  # [-pi/2, pi/2]

         # Heuristic mapping
         az = aolp + (np.pi/2.0)
         tilt = dolp * (np.pi/2.0)

         nx = np.sin(tilt) * np.cos(az)
         ny = np.sin(tilt) * np.sin(az)
         nz = np.cos(tilt)

         # Encode to normal-map colors in RGB, then write as BGR (OpenCV)
         r = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
         g = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
         b = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

         rgb_image[:, :, 2] = r
         rgb_image[:, :, 1] = g
         rgb_image[:, :, 0] = b
    elif way == 23:
          # --- Max / Min / Avg visualization ---
          # Convert to float32 to avoid overflow / clipping
          I0   = polarization_0_deg.astype(np.float32)
          I45  = polarization_45_deg.astype(np.float32)
          I90  = polarization_90_deg.astype(np.float32)
          I135 = polarization_135_deg.astype(np.float32)

          # Stack safely
          stack = np.stack([I0, I45, I90, I135], axis=2)
      
          max_img = np.max(stack, axis=2)
          min_img = np.min(stack, axis=2)
          avg_img = np.mean(stack, axis=2)
      
          max_u8 = normalize_to_u8(max_img)
          min_u8 = normalize_to_u8(min_img)
          avg_u8 = normalize_to_u8(avg_img)
      
          # OpenCV uses BGR
          rgb_image[:, :, 0] = avg_u8   # B
          rgb_image[:, :, 1] = min_u8   # G
          rgb_image[:, :, 2] = max_u8   # R
          print("\n\n\nMIN/MAX/AVG vis ",np.min(max_img), np.max(max_img), np.mean(avg_img) )

    if (contrastValue!=0.0):
           rgb_image = adjust_contrast(rgb_image,contrastValue)

    return rgb_image
