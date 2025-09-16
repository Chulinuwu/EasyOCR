from flask import Flask, request, render_template_string, abort
import os, io, base64, re, sqlite3
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import easyocr

# Google Cloud Vision (optional)
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

app = Flask(__name__)

# Initialize EasyOCR
reader = easyocr.Reader(['th'], gpu=False)
ALLOWLIST = "".join(chr(c) for c in range(0x0E01, 0x0E3B)) + "0123456789"

# Load fonts
try:
    FONT_TH = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", 22)
    FONT_EN = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except:
    FONT_TH = FONT_EN = ImageFont.load_default()

# OCR Parameters
CONSERVATIVE_PARAMS = {
    'contrast_ths': 0.05,
    'adjust_contrast': 0.7,
    'text_threshold': 0.8,
    'low_text': 0.4,
    'link_threshold': 0.5
}

SENSITIVE_PARAMS = {
    'contrast_ths': 0.01,
    'adjust_contrast': 0.7,
    'text_threshold': 0.5,
    'low_text': 0.2,
    'link_threshold': 0.3
}

CONFIDENCE_THRESHOLD = 0.0

def load_authorized_plates():
    """Load authorized plates from labels folder"""
    plates = set()
    
    if not os.path.exists('labels'):
        return plates
    
    for label_file in os.listdir('labels'):
        if label_file.startswith('label_') and label_file.endswith('.txt'):
            with open(os.path.join('labels', label_file), 'r', encoding='utf-8') as f:
                plate_text = f.read().strip()
                if plate_text:
                    plates.add(plate_text)
                    print(f"Loaded authorized plate: '{plate_text}'")
    
    return plates

# Load authorized plates
AUTHORIZED_PLATES = load_authorized_plates()
print(f"Loaded {len(AUTHORIZED_PLATES)} authorized plates")

def normalize_plate_text(text):
    """Normalize plate text for comparison"""
    if not text:
        return ""
    
    # Remove extra whitespace and special characters
    text = re.sub(r'[^\u0E00-\u0E7F0-9A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def is_likely_plate_number(text):
    """Check if text looks like a Thai license plate number"""
    if not text:
        return False
    
    # Remove spaces for analysis
    clean_text = text.replace(' ', '')
    
    # Thai license plate patterns:
    # Pattern 1: [digit][thai][thai][digits] like "1กธ8107" 
    # Pattern 2: [thai][thai][digits] like "กก1234"
    # Pattern 3: [digits][thai][thai][digits] like "12กธ34"
    
    patterns = [
        r'^[0-9][ก-ฮ][ก-ฮ][0-9]+$',  # 1กธ8107
        r'^[ก-ฮ][ก-ฮ][0-9]+$',        # กก1234  
        r'^[0-9]+[ก-ฮ][ก-ฮ][0-9]+$',  # 12กธ34
        r'^[ก-ฮ]+[0-9]+$',            # กง4275
    ]
    
    return any(re.match(pattern, clean_text) for pattern in patterns)

def is_likely_province_name(text):
    """Check if text looks like a Thai province name"""
    if not text:
        return False
    
    # Common Thai province name patterns and keywords
    province_keywords = [
        'กรุงเทพมหานคร', 'กรุงเทพ', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 
        'สมุทรสาคร', 'สมุทรสงคราม', 'นครปฐม', 'อยุธยา', 'ลพบุรี',
        'สิงห์บุรี', 'ชัยนาท', 'สระบุรี', 'ชลบุรี', 'ระยอง', 'จันทบุรี',
        'ตราด', 'ฉะเชิงเทรา', 'ปราจีนบุรี', 'นครนายก', 'สระแก้ว',
        'นครราชสีมา', 'บุรีรัมย์', 'สุรินทร์', 'ศิลาขรรค์', 'มุกดาหาร',
        'นครพนม', 'บึงกาฬ', 'ขอนแก่น', 'อุดรธานี', 'เลย', 'หนองบัวลำภู',
        'อุบลราชธานี', 'ยศธร', 'ชัยภูมิ', 'อำนาจเจริญ', 'หนองคาย',
        'เชียงใหม่', 'ลำพูน', 'ลำปาง', 'แพร่', 'น่าน', 'เชียงราย',
        'แม่ฮ่องสอน', 'พิษณุโลก', 'พิจิตร', 'ตาก', 'สุโขทัย', 'กำแพงเพชร',
        'นครสวรรค์', 'อุทัยธานี', 'เพชรบูรณ์', 'พะเยา', 'เพชรบุรี',
        'ประจวบคีรีขันธ์', 'นครศรีธรรมราช', 'กระบี่', 'พังงา', 'ภูเก็ต',
        'สุราษฎร์ธานี', 'ระนอง', 'ชุมพร', 'สงขลา', 'สตูล', 'ตรัง',
        'พัทลุง', 'ปัตตานี', 'ยะลา', 'นราธิวาส', 'บึงกาฬ'
    ]
    
    # Remove spaces for comparison
    clean_text = text.replace(' ', '')
    
    # Check if it matches any province name (partial match allowed)
    for province in province_keywords:
        if province in clean_text or clean_text in province:
            return True
    
    # Check if it's mostly Thai characters and longer than typical license plate
    thai_chars = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')
    total_chars = sum(1 for c in text if c.isalnum())
    
    if total_chars > 0 and thai_chars / total_chars > 0.7 and len(clean_text) > 6:
        # Likely a province name if mostly Thai and reasonably long
        return True
    
    return False

def smart_license_reconstruction(license_parts):
    """Reconstruct license number from OCR fragments intelligently"""
    if not license_parts:
        return ""
    
    if len(license_parts) == 1:
        # Single part - just clean up spacing
        text = license_parts[0]
        # Remove excessive spaces in Thai license patterns
        # Pattern: "1 ก θ 8107" -> "1กθ 8107"
        
        # First, try to identify Thai license plate pattern and fix spacing
        # Pattern 1: Number + Thai + Thai + Number
        pattern1 = re.match(r'^(\d+)\s*([ก-ฮ])\s*([ก-ฮ])\s*(\d+)$', text.strip())
        if pattern1:
            return f"{pattern1.group(1)}{pattern1.group(2)}{pattern1.group(3)} {pattern1.group(4)}"
        
        # Pattern 2: Thai + Thai + Number  
        pattern2 = re.match(r'^([ก-ฮ])\s*([ก-ฮ])\s*(\d+)$', text.strip())
        if pattern2:
            return f"{pattern2.group(1)}{pattern2.group(2)} {pattern2.group(3)}"
        
        # Pattern 3: Number + Thai + Thai + Number + Number
        pattern3 = re.match(r'^(\d+)\s*([ก-ฮ])\s*([ก-ฮ])\s*(\d+)\s*(\d+)$', text.strip())
        if pattern3:
            return f"{pattern3.group(1)}{pattern3.group(2)}{pattern3.group(3)} {pattern3.group(4)}{pattern3.group(5)}"
        
        # If no pattern matches, just normalize spaces
        return re.sub(r'\s+', ' ', text.strip())
    
    # Multiple parts - combine them intelligently
    combined = ""
    numbers = []
    thai_chars = []
    
    # Separate numbers and Thai characters
    for part in license_parts:
        part = part.strip()
        if part.isdigit():
            numbers.append(part)
        elif any('\u0E00' <= c <= '\u0E7F' for c in part):
            thai_chars.append(part.replace(" ", ""))  # Remove spaces from Thai parts
        else:
            # Mixed content - try to separate
            thai_part = re.sub(r'[^\u0E00-\u0E7F]', '', part)
            num_part = re.sub(r'[^\d]', '', part)
            if thai_part:
                thai_chars.append(thai_part)
            if num_part:
                numbers.append(num_part)
    
    # Reconstruct based on Thai license plate patterns
    if numbers and thai_chars:
        # Most common pattern: [number][thai][thai] [numbers]
        thai_combined = "".join(thai_chars)
        if len(numbers) >= 2:
            # Pattern: 1กθ 8107
            combined = f"{numbers[0]}{thai_combined} {''.join(numbers[1:])}"
        else:
            # Pattern: 1กθ8107 or กθ1234
            combined = f"{numbers[0] if numbers else ''}{thai_combined}{''.join(numbers[1:]) if len(numbers) > 1 else ''}"
    elif numbers:
        combined = "".join(numbers)
    elif thai_chars:
        combined = "".join(thai_chars)
    else:
        combined = " ".join(license_parts)
    
    return combined.strip()

def combine_ocr_fragments(results):
    """Combine OCR fragments that might belong to the same license plate - Enhanced version"""
    if len(results) <= 1:
        return results
    
    # Calculate average text height for better grouping
    avg_height = sum((max(pt[1] for pt in bbox) - min(pt[1] for pt in bbox)) for bbox, _, _ in results) / len(results)
    
    # Group results by spatial proximity
    groups = []
    
    for bbox, text, conf in results:
        center_y = sum(pt[1] for pt in bbox) / 4
        center_x = sum(pt[0] for pt in bbox) / 4
        
        # Calculate bounding box dimensions
        bbox_width = max(pt[0] for pt in bbox) - min(pt[0] for pt in bbox)
        bbox_height = max(pt[1] for pt in bbox) - min(pt[1] for pt in bbox)
        
        # Find group with similar spatial position
        found_group = False
        for group in groups:
            group_center_y = sum(item[4] for item in group) / len(group)
            group_center_x = sum(item[3] for item in group) / len(group)
            
            # Enhanced grouping criteria:
            # 1. Y-position similarity (same horizontal line)
            y_threshold = max(avg_height * 1.5, 40)  # Adaptive threshold
            y_close = abs(center_y - group_center_y) < y_threshold
            
            # 2. X-position proximity (not too far apart horizontally)
            x_threshold = max(bbox_width * 3, 150)  # Maximum horizontal distance
            x_close = abs(center_x - group_center_x) < x_threshold
            
            # 3. Check if they could be part of the same license plate area
            # License plates usually have consistent height
            group_heights = [item[6] for item in group]  # bbox_height stored at index 6
            avg_group_height = sum(group_heights) / len(group_heights)
            height_similar = abs(bbox_height - avg_group_height) < avg_height * 0.5
            
            if y_close and (x_close or height_similar):
                group.append((bbox, text, conf, center_x, center_y, bbox_width, bbox_height))
                found_group = True
                break
        
        if not found_group:
            groups.append([(bbox, text, conf, center_x, center_y, bbox_width, bbox_height)])
    
    # Process each group to create combined results
    combined_results = []
    
    for group in groups:
        if len(group) == 1:
            bbox, text, conf, _, _, _, _ = group[0]
            combined_results.append((bbox, text, conf))
            continue
        
        # Sort group by X position (left to right)
        group.sort(key=lambda x: x[3])  # Sort by center_x
        
        # Analyze the group content
        texts = [item[1].strip() for item in group]
        
        # Separate different types of content
        license_numbers = []  # Text that looks like license plate numbers
        province_names = []   # Text that looks like province names
        other_text = []       # Other text
        
        for text in texts:
            if is_likely_plate_number(text):
                license_numbers.append(text)
            elif is_likely_province_name(text):
                province_names.append(text)
            else:
                # Try to determine if it's more likely a number or province
                if any(c.isdigit() for c in text):
                    license_numbers.append(text)
                else:
                    province_names.append(text)
        
        # Create combined text based on content analysis
        combined_text_options = []
        
        # Option 1: Simple left-to-right combination
        simple_combo = " ".join(texts)
        combined_text_options.append(simple_combo)
        
        # Option 2: License number + Province (Thai license plate format)
        if license_numbers and province_names:
            # Try different arrangements
            license_part = " ".join(license_numbers)
            province_part = " ".join(province_names)
            
            # Format: "1กθ 8107 กรุงเทพมหานคร"
            combined_text_options.append(f"{license_part} {province_part}")
            
            # Also try: "1กθ 8107" (just the license number for matching)
            combined_text_options.append(license_part)
        
        # Option 3: Try to reconstruct Thai license plate patterns
        all_text_no_space = "".join(text.replace(" ", "") for text in texts)
        if len(all_text_no_space) > 3:
            combined_text_options.append(all_text_no_space)
        
        # Choose the best combination
        best_combo = simple_combo
        
        # Prefer combinations that match known license plate patterns
        for combo in combined_text_options:
            if is_likely_plate_number(combo.split()[0] if combo.split() else combo):
                best_combo = combo
                break
        
        # Create combined bounding box that covers all fragments
        all_points = []
        total_conf = 0
        for bbox, text, conf, _, _, _, _ in group:
            all_points.extend(bbox)
            total_conf += conf
        
        # Calculate combined bounding box
        min_x = min(pt[0] for pt in all_points)
        max_x = max(pt[0] for pt in all_points)
        min_y = min(pt[1] for pt in all_points)
        max_y = max(pt[1] for pt in all_points)
        
        combined_bbox = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        combined_conf = total_conf / len(group)
        
        combined_results.append((combined_bbox, best_combo, combined_conf))
        
        # If we have both license number and province, also add just the license number
        # This helps with matching against authorized plates that might not include province
        if license_numbers and province_names and len(license_numbers) == 1:
            license_only = license_numbers[0]
            if license_only != best_combo:  # Avoid duplicates
                # Create a smaller bounding box for just the license number part
                license_bbox = None
                for bbox, text, conf, _, _, _, _ in group:
                    if text.strip() == license_only:
                        license_bbox = bbox
                        break
                
                if not license_bbox:
                    license_bbox = combined_bbox  # Fallback to combined bbox
                
                combined_results.append((license_bbox, license_only, combined_conf))
    
    return combined_results

def check_authorization(detected_text):
    """Check if detected text matches any authorized plate"""
    normalized_detected = normalize_plate_text(detected_text)
    
    # Try exact match first
    for authorized_plate in AUTHORIZED_PLATES:
        normalized_authorized = normalize_plate_text(authorized_plate)
        
        if normalized_detected == normalized_authorized:
            return {
                "status": "PASS",
                "matched_plate": authorized_plate,
                "match_type": "Exact match"
            }
    
    # Try fuzzy match (ignore spaces)
    detected_no_space = normalized_detected.replace(' ', '')
    for authorized_plate in AUTHORIZED_PLATES:
        authorized_no_space = normalize_plate_text(authorized_plate).replace(' ', '')
        
        if detected_no_space == authorized_no_space:
            return {
                "status": "PASS", 
                "matched_plate": authorized_plate,
                "match_type": "Space-insensitive match"
            }
    
    # Try character-level fuzzy matching for OCR errors
    detected_chars = set(char for char in normalized_detected if char != ' ')
    for authorized_plate in AUTHORIZED_PLATES:
        authorized_chars = set(char for char in normalize_plate_text(authorized_plate) if char != ' ')
        
        # Check if most characters match
        if len(detected_chars & authorized_chars) >= min(len(detected_chars), len(authorized_chars)) * 0.8:
            return {
                "status": "PASS",
                "matched_plate": authorized_plate,
                "match_type": "Character-level fuzzy match"
            }
    
    # Try partial match (for fragmented OCR)
    for authorized_plate in AUTHORIZED_PLATES:
        normalized_authorized = normalize_plate_text(authorized_plate)
        if (normalized_detected in normalized_authorized or 
            normalized_authorized in normalized_detected):
            return {
                "status": "PARTIAL",
                "matched_plate": authorized_plate, 
                "match_type": "Partial match"
            }
    
    # Try reverse order match (for OCR reading direction errors)
    detected_parts = normalized_detected.split()
    if len(detected_parts) > 1:
        reversed_text = " ".join(reversed(detected_parts))
        for authorized_plate in AUTHORIZED_PLATES:
            normalized_authorized = normalize_plate_text(authorized_plate)
            if reversed_text == normalized_authorized:
                return {
                    "status": "PASS",
                    "matched_plate": authorized_plate,
                    "match_type": "Reverse order match"
                }
    
    return {
        "status": "FAIL",
        "matched_plate": None,
        "match_type": "Not authorized"
    }

def pil_to_base64(img, quality=90):
    """Convert PIL image to base64"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def preprocess_image(img):
    """Enhanced image preprocessing"""
    img2 = ImageEnhance.Contrast(img).enhance(1.25)
    img2 = img2.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return img2

def draw_mixed_text(draw, pos, text, fill=(255,255,255)):
    """Draw text with mixed Thai/English fonts"""
    x, y = pos
    for ch in text:
        font = FONT_TH if '\u0E00' <= ch <= '\u0E7F' else FONT_EN
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch)

def draw_annotations(img, results, auth_results):
    """Draw bounding boxes and annotations"""
    draw = ImageDraw.Draw(img)
    
    for i, (bbox, text, conf) in enumerate(results):
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        auth = auth_results[i] if i < len(auth_results) else {"status": "UNKNOWN"}
        
        # Color based on status
        if auth["status"] == "PASS":
            color = (0, 255, 0)  # Green
        elif auth["status"] == "PARTIAL":
            color = (255, 165, 0)  # Orange
        else:
            color = (255, 0, 0)  # Red
        
        # Draw bounding box
        pts = [(int(x), int(y)) for x, y in bbox]
        draw.polygon(pts, outline=color, width=3)
        
        # Draw text and status
        status_text = f"{text} - {auth['status']}"
        if auth["status"] in ["PASS", "PARTIAL"] and auth["matched_plate"]:
            status_text += f" ({auth['matched_plate']})"
        
        draw_mixed_text(draw, (pts[0][0], max(pts[0][1]-35, 0)), status_text, fill=color)

def run_easyocr(img, use_sensitive=False):
    """Run EasyOCR with specified parameters"""
    img_processed = preprocess_image(img)
    arr = np.array(img_processed)
    
    params = SENSITIVE_PARAMS if use_sensitive else CONSERVATIVE_PARAMS
    
    results = reader.readtext(
        arr,
        detail=1,
        paragraph=False,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        rotation_info=[0, 90, -90],
        **params
    )
    
    return results

def run_google_ocr(img):
    """Run Google Cloud Vision OCR"""
    if not GOOGLE_VISION_AVAILABLE:
        return []
    
    credentials_files = ['ggcloud-key.json', 'google-credentials.json', 'service-account.json']
    
    try:
        credentials_path = None
        for file in credentials_files:
            if os.path.exists(file):
                credentials_path = file
                break
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            client = vision.ImageAnnotatorClient()
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        content = buf.getvalue()
        
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        
        if response.error.message:
            print(f'Google Vision API Error: {response.error.message}')
            return []
        
        results = []
        if response.text_annotations:
            for annotation in response.text_annotations[1:]:
                vertices = annotation.bounding_poly.vertices
                bbox = [(vertex.x, vertex.y) for vertex in vertices]
                text = annotation.description
                confidence = 0.9
                results.append((bbox, text, confidence))
        
        return results
    
    except Exception as e:
        print(f"Google OCR Error: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        if not f: 
            return abort(400, "No file uploaded")
        
        ocr_method = request.form.get("ocr_method", "easyocr")
        use_sensitive = True  # Always use sensitive parameters
        
        img = Image.open(f.stream).convert("RGB")
        
        # Run OCR
        if ocr_method == "google" and GOOGLE_VISION_AVAILABLE:
            results = run_google_ocr(img)
            ocr_info = "Google Cloud Vision API + Sensitive params + Fragment combination"
        else:
            results = run_easyocr(img, use_sensitive)
            ocr_info = "EasyOCR (Sensitive parameters + Fragment combination)"
        
        # Filter by confidence
        filtered_results = [(bbox, text, conf) for bbox, text, conf in results 
                          if conf >= CONFIDENCE_THRESHOLD]
        
        # Combine fragments (always enabled)
        filtered_results = combine_ocr_fragments(filtered_results)
        
        # Check authorization
        auth_results = []
        for _, text, conf in filtered_results:
            auth_result = check_authorization(text)
            auth_results.append(auth_result)
        
        # Prepare results for display
        result_details = []
        for i, (bbox, text, conf) in enumerate(filtered_results):
            auth = auth_results[i]
            result_details.append({
                'text': text,
                'confidence': conf,
                'status': auth['status']
            })
        
        # Draw annotations
        img_annotated = img.copy()
        draw_annotations(img_annotated, filtered_results, auth_results)
        
        return render_template_string(HTML_TEMPLATE, 
                                    result_b64=pil_to_base64(img_annotated),
                                    results=result_details,
                                    ocr_info=ocr_info,
                                    authorized_plates=sorted(AUTHORIZED_PLATES),
                                    google_available=GOOGLE_VISION_AVAILABLE)
    
    return render_template_string(HTML_TEMPLATE, 
                                authorized_plates=sorted(AUTHORIZED_PLATES),
                                google_available=GOOGLE_VISION_AVAILABLE)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>License Plate OCR System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-form { margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100%; max-width: 300px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 30px; }
        .result-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .result-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .result-table th, .result-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .result-table th { background: #f8f9fa; }
        .status-pass { color: #28a745; font-weight: bold; }
        .status-partial { color: #ffc107; font-weight: bold; }
        .status-fail { color: #dc3545; font-weight: bold; }
        .info-box { background: #e9ecef; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .authorized-plates { background: #d4edda; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .plate-list { display: flex; flex-wrap: wrap; gap: 10px; }
        .plate-item { background: #fff; padding: 5px 10px; border: 1px solid #c3e6cb; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 License Plate OCR System</h1>
        
        <div class="authorized-plates">
            <h3>Authorized License Plates ({{ authorized_plates|length }})</h3>
            <div class="plate-list">
                {% for plate in authorized_plates %}
                <div class="plate-item">{{ plate }}</div>
                {% endfor %}
            </div>
        </div>
        
        <form method="post" enctype="multipart/form-data" class="upload-form">
            <h3>Upload License Plate Image</h3>
            
            <div class="form-group">
                <label for="image">Select Image:</label>
                <input type="file" name="image" id="image" accept="image/*" required>
            </div>
            
            <div class="form-group">
                <label for="ocr_method">OCR Method:</label>
                <select name="ocr_method" id="ocr_method">
                    <option value="easyocr">EasyOCR</option>
                    {% if google_available %}
                    <option value="google">Google Cloud Vision API</option>
                    {% endif %}
                </select>
            </div>
            
            <button type="submit">Analyze License Plate</button>
        </form>
        
        {% if results %}
        <div class="result">
            <div class="info-box">
                <strong>OCR Engine:</strong> {{ ocr_info }}<br>
                <strong>Results Found:</strong> {{ results|length }} text regions
            </div>
            
            <h3>Annotated Image</h3>
            <img src="data:image/jpeg;base64,{{ result_b64 }}" class="result-image" alt="Annotated result">
            
            <h3>Detection Results</h3>
            <table class="result-table">
                <thead>
                    <tr>
                        <th>Detected Text</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr>
                        <td><strong>{{ result.text }}</strong></td>
                        <td>{{ "%.1f%%"|format(result.confidence * 100) }}</td>
                        <td class="status-{{ result.status.lower() }}">{{ result.status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if not google_available %}
        <div class="info-box">
            <strong>Note:</strong> Google Cloud Vision API is not available.
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting License Plate OCR System...")
    print(f"EasyOCR: Ready")
    print(f"Google Cloud Vision: {'Available' if GOOGLE_VISION_AVAILABLE else 'Not Available'}")
    print(f"Authorized plates loaded: {len(AUTHORIZED_PLATES)}")
    app.run(host="0.0.0.0", port=5000, debug=True)