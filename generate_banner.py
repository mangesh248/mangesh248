import os
import sys
import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def load_and_preprocess_photo(img_path, target_width=200, target_height=226):
    """
    Loads photo.jpg, crops head and shoulders, resizes to target grid (200x226),
    enhances contrast (1.3x, autocontrast cutoff=1, UnsharpMask radius=3, percent=140).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # Center crop to 300:340 aspect ratio (~ 1 : 1.1333)
    target_aspect = target_width / target_height
    curr_aspect = w / h
    if curr_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
        
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    return img

def segment_background_dark_mode(img):
    """
    Segment background out for dark mode so dots draw the lit subject on the panel.
    Threshold on colour distance from background corner colors, binary closing, fill holes,
    keep largest component.
    """
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    # Estimate background color from top-left and top-right corners
    bg_color = (np.mean(arr[0:20, 0:20], axis=(0,1)) + np.mean(arr[0:20, w-20:w], axis=(0,1))) / 2.0
    
    # Distance from background color
    diff = arr - bg_color
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Thresholding
    threshold = max(25.0, np.percentile(dist, 25))
    mask = dist > threshold
    
    # Morphological closing and hole filling
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    
    # Keep largest connected component
    labeled, num_features = ndi.label(mask)
    if num_features > 0:
        sizes = ndi.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        mask = (labeled == largest_label)
        
    return mask

def floyd_steinberg_dither(img_gray, mask=None, invert_for_light_mode=False):
    """
    1-bit Floyd-Steinberg dither in serpentine order.
    Returns boolean array of dot locations.
    """
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    if invert_for_light_mode:
        arr = 1.0 - arr
        
    h, w = arr.shape
    dots = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        x_range = range(w) if (y % 2 == 0) else range(w - 1, -1, -1)
        for x in x_range:
            old_val = arr[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            
            # Mask edge check: if mask is provided and pixel is outside mask, clear dot
            if mask is not None and not mask[y, x]:
                new_val = 0.0
                err = 0.0
            else:
                err = old_val - new_val
                dots[y, x] = (new_val == 1.0)
                
            # Distribute error in serpentine order
            dir_x = 1 if (y % 2 == 0) else -1
            if 0 <= x + dir_x < w:
                arr[y, x + dir_x] += err * (7.0 / 16.0)
            if y + 1 < h:
                if 0 <= x - dir_x < w:
                    arr[y + 1, x - dir_x] += err * (3.0 / 16.0)
                arr[y + 1, x] += err * (5.0 / 16.0)
                if 0 <= x + dir_x < w:
                    arr[y + 1, x + dir_x] += err * (1.0 / 16.0)
                    
    return dots

def generate_logo_point_clouds(num_points=900, cx=240, cy=330, scale=120):
    """
    Generate 3 distinct logo point clouds (~900 points each) centered at (cx, cy).
    1. Python (two interlocking rectangles/loops)
    2. React (three ellipses + central atom)
    3. Code brackets </>
    """
    np.random.seed(42)
    
    # Logo 1: Python logo shape
    logo1 = []
    while len(logo1) < num_points:
        t = np.random.uniform(0, 2 * math.pi)
        r = scale * 0.7 * math.sqrt(np.random.uniform(0.2, 1.0))
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t) * 0.8
        # Add shape cutouts for python style
        if not (-scale*0.2 < (x-cx) < scale*0.2 and -scale*0.1 < (y-cy) < scale*0.1):
            logo1.append((x, y))
    logo1 = np.array(logo1[:num_points])
    
    # Logo 2: React logo shape (three ellipses rotated by 0, 60, 120 deg)
    logo2 = []
    while len(logo2) < num_points:
        choice = np.random.randint(0, 4)
        if choice == 0:  # center nucleus
            r = scale * 0.2 * math.sqrt(np.random.uniform(0, 1.0))
            theta = np.random.uniform(0, 2 * math.pi)
            logo2.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        else:
            angle_deg = (choice - 1) * 60
            angle_rad = math.radians(angle_deg)
            t = np.random.uniform(0, 2 * math.pi)
            a = scale * 0.9
            b = scale * 0.25
            ex = a * math.cos(t)
            ey = b * math.sin(t)
            rx = ex * math.cos(angle_rad) - ey * math.sin(angle_rad)
            ry = ex * math.sin(angle_rad) + ey * math.cos(angle_rad)
            logo2.append((cx + rx, cy + ry))
    logo2 = np.array(logo2[:num_points])
    
    # Logo 3: Code brackets </>
    logo3 = []
    while len(logo3) < num_points:
        part = np.random.randint(0, 3)
        if part == 0:  # <
            t = np.random.uniform(-1, 1)
            x = cx - scale*0.45 + abs(t) * scale*0.3
            y = cy + t * scale*0.6
            logo3.append((x, y))
        elif part == 1:  # >
            t = np.random.uniform(-1, 1)
            x = cx + scale*0.45 - abs(t) * scale*0.3
            y = cy + t * scale*0.6
            logo3.append((x, y))
        else:  # /
            t = np.random.uniform(-1, 1)
            x = cx + t * scale*0.25
            y = cy - t * scale*0.65
            logo3.append((x, y))
    logo3 = np.array(logo3[:num_points])
    
    return logo1, logo2, logo3

def match_point_clouds(p1, p2):
    """
    Optimal transport matching using linear sum assignment.
    """
    dist_matrix = cdist(p1, p2, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return p2[col_ind]

def generate_svg(dark_mode=True):
    # Palette
    if dark_mode:
        bg_color = "#0A101F"
        portrait_color = "#A78BFA"
        ui_color = "#22D3EE"
        accent_color = "#10B981"
        text_color = "#94A3B8"
        title_color = "#22D3EE"
        border_color = "#1E293B"
        pill_bg = "#1E293B"
        pill_text = "#A78BFA"
    else:
        bg_color = "#F8FAFC"
        portrait_color = "#7C3AED"
        ui_color = "#0891B2"
        accent_color = "#10B981"
        text_color = "#334155"
        title_color = "#0891B2"
        border_color = "#E2E8F0"
        pill_bg = "#E2E8F0"
        pill_text = "#7C3AED"
        
    # Load and process photo
    img = load_and_preprocess_photo("photo.jpg", target_width=200, target_height=226)
    img_gray = img.convert("L")
    
    if dark_mode:
        mask = segment_background_dark_mode(img)
        dots_mask = floyd_steinberg_dither(img_gray, mask=mask, invert_for_light_mode=False)
    else:
        dots_mask = floyd_steinberg_dither(img_gray, mask=None, invert_for_light_mode=True)
        
    # Extract dot coordinates in portrait area (frame bounds: x in [50, 430], y in [130, 560])
    # Mapping grid (200, 226) -> frame size (380, 430)
    ys, xs = np.where(dots_mask)
    dot_coords = []
    for x, y in zip(xs, ys):
        canvas_x = 50 + x * (380.0 / 200.0)
        canvas_y = 130 + y * (430.0 / 226.0)
        dot_coords.append((canvas_x, canvas_y))
        
    dot_coords = np.array(dot_coords)
    num_portrait_dots = len(dot_coords)
    print(f"['{'Dark' if dark_mode else 'Light'} mode'] Extracted {num_portrait_dots} portrait dots.")
    
    # 1. Portrait Layer - Intro animation & Drift bands
    # Add per-dot noise (sigma=4) before grouping to avoid square grid trap
    noisy_y = dot_coords[:, 1] + np.random.normal(0, 4.0, size=num_portrait_dots)
    # Sort into ~94 drift bands
    num_bands = min(94, num_portrait_dots)
    band_indices = np.array_split(np.argsort(noisy_y), num_bands)
    
    # Also assign intro groups (~60 interleaved random groups across whole portrait)
    intro_groups = np.random.randint(0, 60, size=num_portrait_dots)
    
    # Group path strings by band for efficiency
    portrait_paths = []
    for band_idx, indices in enumerate(band_indices):
        d_parts = [f"M{dot_coords[i, 0]:.1f},{dot_coords[i, 1]:.1f}h1.3" for i in indices]
        d_str = "".join(d_parts)
        
        # Calculate drift offset (~42% toward center (240, 345))
        mean_x = np.mean(dot_coords[indices, 0])
        mean_y = np.mean(dot_coords[indices, 1])
        dx = (240.0 - mean_x) * 0.42
        dy = (345.0 - mean_y) * 0.42
        
        # Loop keyframes (14.2s total: 0-3s hold, 3-4.3s fade/drift, 4.3-12.9s hidden, 12.9-14.2s return)
        key_times = "0;0.211;0.302;0.908;1"
        opacity_vals = "1;1;0;0;1"
        translate_vals = f"0,0;0,0;{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};0,0"
        
        portrait_paths.append(f'''<path d="{d_str}" stroke="{portrait_color}" stroke-width="1.3" shape-rendering="crispEdges">
      <animate attributeName="opacity" values="{opacity_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite" />
      <animateTransform attributeName="transform" type="translate" values="{translate_vals}" keyTimes="{key_times}" dur="14.2s" repeatCount="indefinite" />
    </path>''')
    
    # 2. Travellers Layer (~900 dots morphing between 3 logos)
    logo1, logo2, logo3 = generate_logo_point_clouds(num_points=900, cx=240, cy=345, scale=110)
    logo2_matched = match_point_clouds(logo1, logo2)
    logo3_matched = match_point_clouds(logo2_matched, logo3)
    logo1_ret_matched = match_point_clouds(logo3_matched, logo1)
    
    traveller_elements = []
    # KeyTimes for 14.2s cycle:
    # 0.0s (0) to 3.0s (0.211): Hidden (opacity 0)
    # 3.0s (0.211) to 4.3s (0.302): Transition Portrait -> Logo1 (opacity fades in 0 -> 1)
    # 4.3s (0.302) to 6.3s (0.443): Logo1 hold
    # 6.3s (0.443) to 7.6s (0.535): Transition Logo1 -> Logo2
    # 7.6s (0.535) to 9.6s (0.676): Logo2 hold
    # 9.6s (0.676) to 10.9s (0.767): Transition Logo2 -> Logo3
    # 10.9s (0.767) to 12.9s (0.908): Logo3 hold
    # 12.9s (0.908) to 14.2s (1.000): Transition Logo3 -> Portrait (opacity fades out 1 -> 0)
    t_times = "0;0.211;0.302;0.443;0.535;0.676;0.767;0.908;1"
    t_opacity = "0;0;1;1;1;1;1;1;0"
    
    for i in range(len(logo1)):
        x1, y1 = logo1[i]
        x2, y2 = logo2_matched[i]
        x3, y3 = logo3_matched[i]
        x1r, y1r = logo1_ret_matched[i]
        
        x_vals = f"{x1:.1f};{x1:.1f};{x1:.1f};{x1:.1f};{x2:.1f};{x2:.1f};{x3:.1f};{x3:.1f};{x1r:.1f}"
        y_vals = f"{y1:.1f};{y1:.1f};{y1:.1f};{y1:.1f};{y2:.1f};{y2:.1f};{y3:.1f};{y3:.1f};{y1r:.1f}"
        
        traveller_elements.append(f'''<circle r="1.6" fill="{portrait_color}">
      <animate attributeName="cx" values="{x_vals}" keyTimes="{t_times}" dur="14.2s" repeatCount="indefinite" />
      <animate attributeName="cy" values="{y_vals}" keyTimes="{t_times}" dur="14.2s" repeatCount="indefinite" />
      <animate attributeName="opacity" values="{t_opacity}" keyTimes="{t_times}" dur="14.2s" repeatCount="indefinite" />
    </circle>''')

    # Info Panel Rows (Right side ~58%, starting x=500, w=630)
    info_rows = [
        ("Subject", "Mangesh Wagh"),
        ("Role", "Full-Stack Dev · DevOps"),
        ("Origin", "Aurangabad, India"),
        ("Education", "CSE Undergrad (GECA '28)"),
        ("Status", "Building + Learning + Shipping"),
        ("ToolChain", "Python, C++, C, JavaScript"),
        ("Core.Frontend", "React, Next.js, Tailwind CSS"),
        ("Core.Backend", "Node.js, Express.js"),
        ("Core.Database", "PostgreSQL, MySQL"),
        ("Core.Infra", "Docker, Kubernetes, CI/CD"),
        ("Grid.Mail", "wmangesh91@gmail.com"),
        ("Grid.LinkedIn", "linkedin.com/in/mangesh-wagh-"),
        ("Grid.GitHub", "mangesh248"),
        ("Grid.Codeforces", "mangeshh223"),
        ("Grid.LeetCode", "mangesh24891")
    ]
    
    rows_svg = []
    start_y = 150
    for idx, (label, val) in enumerate(info_rows):
        y_pos = start_y + idx * 27
        # Compute dotted leaders dynamically
        dot_count = max(5, int((520 - (len(label)*8.5 + len(val)*7.5)) / 6.0))
        dots_leader = "." * dot_count
        
        rows_svg.append(f'''
      <g transform="translate(510, {y_pos})">
        <text x="0" y="0" font-family="'Courier New', monospace" font-size="14" font-weight="600" fill="{title_color}">{label}</text>
        <text x="210" y="0" font-family="'Courier New', monospace" font-size="12" fill="{text_color}" opacity="0.4">{dots_leader}</text>
        <text x="590" y="0" font-family="'Courier New', monospace" font-size="14" font-weight="500" fill="{text_color}" text-anchor="end" textLength="{max(120, len(val)*8.5):.0f}" lengthAdjust="spacingAndGlyphs">{val}</text>
      </g>''')

    # Assemble complete SVG
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
  <defs>
    <style>
      @keyframes pulse {{
        0% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.4; transform: scale(1.15); }}
        100% {{ opacity: 1; transform: scale(1); }}
      }}
      .live-dot {{ animation: pulse 2s infinite ease-in-out; transform-origin: center; }}
    </style>
  </defs>

  <!-- Window Background -->
  <rect x="0" y="0" width="1180" height="610" rx="12" fill="{bg_color}" stroke="{border_color}" stroke-width="2" />

  <!-- Title Bar -->
  <g transform="translate(0, 0)">
    <rect x="0" y="0" width="1180" height="48" rx="12" fill="{bg_color}" />
    <line x1="0" y1="48" x2="1180" y2="48" stroke="{border_color}" stroke-width="1.5" />
    
    <!-- Window buttons -->
    <circle cx="28" cy="24" r="6" fill="#EF4444" />
    <circle cx="48" cy="24" r="6" fill="#F59E0B" />
    <circle cx="68" cy="24" r="6" fill="#10B981" />
    
    <!-- VISUAL.MAP title -->
    <text x="100" y="29" font-family="'Courier New', monospace" font-size="13" font-weight="700" fill="{title_color}" letter-spacing="1">VISUAL.MAP</text>
    
    <!-- Right side: profile.sh - -live -->
    <text x="1150" y="29" font-family="'Courier New', monospace" font-size="13" font-weight="600" fill="{text_color}" text-anchor="end">profile.sh --live</text>
  </g>

  <!-- Left Column: Portrait Frame (~38% width -> x=30 to 450) -->
  <g id="portrait-area">
    <rect x="35" y="70" width="410" height="510" rx="10" fill="none" stroke="{border_color}" stroke-width="1.5" />
    
    <!-- Header inside left frame: LIVE badge & handle pill -->
    <g transform="translate(55, 95)">
      <!-- LIVE pulsing red badge -->
      <circle cx="8" cy="-4" r="5" fill="#EF4444" class="live-dot" />
      <text x="20" y="0" font-family="'Courier New', monospace" font-size="12" font-weight="700" fill="#EF4444">LIVE</text>
      
      <!-- Coloured pill with handle -->
      <rect x="250" y="-14" width="110" height="22" rx="11" fill="{pill_bg}" />
      <text x="305" y="2" font-family="'Courier New', monospace" font-size="13" font-weight="700" fill="{pill_text}" text-anchor="middle">@mangesh248</text>
    </g>

    <!-- Portrait & Swarm Area -->
    <g id="portrait-dots">
      {"".join(portrait_paths)}
    </g>
    <g id="travellers-dots">
      {"".join(traveller_elements)}
    </g>
  </g>

  <!-- Right Column: SYSTEM.INFO Readout (~58% width -> x=480 to 1145) -->
  <g id="info-panel">
    <!-- Header -->
    <text x="510" y="105" font-family="'Courier New', monospace" font-size="15" font-weight="700" fill="{title_color}" letter-spacing="1.5">// SYSTEM.INFO</text>
    <line x1="510" y1="118" x2="1135" y2="118" stroke="{border_color}" stroke-width="1" />
    
    <!-- Data Rows -->
    {"".join(rows_svg)}
  </g>
</svg>'''

    return svg_content

if __name__ == "__main__":
    print("Generating dark.svg...")
    dark_svg = generate_svg(dark_mode=True)
    with open("dark.svg", "w", encoding="utf-8") as f:
        f.write(dark_svg)
    print("Saved dark.svg successfully.")

    print("Generating light.svg...")
    light_svg = generate_svg(dark_mode=False)
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(light_svg)
    print("Saved light.svg successfully.")
