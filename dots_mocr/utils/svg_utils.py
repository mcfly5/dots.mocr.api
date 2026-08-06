import re
import cairosvg
from PIL import Image, ImageDraw, ImageFont 

def fix_svg(svg: str) -> str:
    """Repair incomplete SVG tags."""
    # 1) Targeted fix: a trailing <path d="... where the d attribute is unclosed.
    if re.search(r'(<path\b[^>]*\bd="[^">]*$)', svg):
        svg += '">'

    # 2) Remove a truncated tag at the end.
    svg = re.sub(r'<[^>]*$', '', svg)

    # 3) Scan in order with a stack to close any unclosed tags.
    stack = []
    TAG_RE = re.compile(r'</?\s*([a-zA-Z][\w:-]*)\b[^>]*?/?>')
    for m in TAG_RE.finditer(svg):
        name = m.group(1)
        token = m.group(0)
        is_close = token.lstrip().startswith("</")
        is_self_close = token.rstrip().endswith("/>")
        
        if is_self_close:
            continue
        if not is_close:
            stack.append(name)
        else:
            if name in stack[::-1]:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack and stack[-1] == name:
                    stack.pop()
    
    # 4) Close any remaining unclosed tags.
    while stack:
        svg += f'</{stack.pop()}>'

    return svg


def extract_svg_from_response(response: str):
    """Extract SVG content from the model response, returns (svg_content, success)."""
    response = response.replace("svg:", "").strip()

    # Try to match a complete <svg>...</svg>.
    svg_match = re.search(r'<svg[^>]*>(.*?)</svg>', response, re.DOTALL)
    if svg_match:
        return svg_match.group(0), True

    # Try to match an incomplete SVG.
    svg_match = re.search(r'<svg[^>]*>.*', response, re.DOTALL)
    if svg_match:
        return fix_svg(svg_match.group(0)), True

    return None, False


def svg_to_png(svg_content: str, output_path: str, width: int = 1024, height: int = 1024):
    """Convert SVG to a PNG image."""
    import cairosvg
    try:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            output_width=width,
            output_height=height,
            background_color='white'
        )
        return True, None
    except Exception as e:
        return False, str(e)

def _add_label(image: Image.Image, label: str, font_size: int = 24) -> Image.Image:
    """Add a label to the top-right corner of the image."""
    draw = ImageDraw.Draw(image)

    # Load font (prefer bold).
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()  # fall back to the default font if not found

    # Compute text position (top-right corner).
    padding = 10
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = image.width - text_width - padding * 2  # align right
    y = padding  # align top

    # Draw a translucent black background.
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)  # black, 180 is the opacity
    )

    # Merge the background layer.
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)

    # Draw the white text.
    draw = ImageDraw.Draw(image)
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
    
    return image


def create_comparison_image(original_image, rendered_image, gap=10, 
                            top_label: str = "Origin", 
                            bottom_label: str = "Generated"):
    """
    Create a comparison image: original on top, rendered below, with labels.

    Args:
        original_image: PIL Image, the original image.
        rendered_image: PIL Image or path, the rendered image.
        gap: Gap height between the two images (pixels).
        top_label: Top image label, defaults to "Origin".
        bottom_label: Bottom image label, defaults to "Generated".

    Returns:
        PIL Image: The stitched comparison image.
    """
    if isinstance(rendered_image, str):
        rendered_image = Image.open(rendered_image)

    # Unify width, scaling proportionally.
    target_width = max(original_image.width, rendered_image.width)

    # Scale the original image.
    if original_image.width != target_width:
        scale = target_width / original_image.width
        new_height = int(original_image.height * scale)
        original_image = original_image.resize((target_width, new_height), Image.LANCZOS)

    # Scale the rendered image.
    if rendered_image.width != target_width:
        scale = target_width / rendered_image.width
        new_height = int(rendered_image.height * scale)
        rendered_image = rendered_image.resize((target_width, new_height), Image.LANCZOS)

    # Convert to RGBA mode so labels can be added.
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    if rendered_image.mode != 'RGBA':
        rendered_image = rendered_image.convert('RGBA')

    # Add labels.
    original_image = _add_label(original_image, top_label)
    rendered_image = _add_label(rendered_image, bottom_label)

    # Convert back to RGB mode for stitching.
    if original_image.mode == 'RGBA':
        bg = Image.new('RGB', original_image.size, (255, 255, 255))
        bg.paste(original_image, mask=original_image.split()[3])
        original_image = bg

    if rendered_image.mode == 'RGBA':
        bg = Image.new('RGB', rendered_image.size, (255, 255, 255))
        bg.paste(rendered_image, mask=rendered_image.split()[3])
        rendered_image = bg

    # Compute the stitched dimensions.
    total_height = original_image.height + gap + rendered_image.height

    # Create a blank canvas.
    comparison = Image.new('RGB', (target_width, total_height), (255, 255, 255))

    # Paste the two images: original on top, rendered below.
    comparison.paste(original_image, (0, 0))
    comparison.paste(rendered_image, (0, original_image.height + gap))
    
    return comparison