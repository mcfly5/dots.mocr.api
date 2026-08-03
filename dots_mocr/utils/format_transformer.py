import os
import sys
import json
import logging
import re
from pathlib import Path

from PIL import Image
from dots_mocr.utils.image_utils import PILimage_to_base64

logger = logging.getLogger("uvicorn.error")

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def resolve_describe_script(describe_script: str) -> Path:
    """Resolve a describe_script value to an allowed script path.

    Only scripts inside the repository's `scripts/` directory may be executed;
    describe_script comes from the API request, so anything else would let a
    client run arbitrary Python files on the server.

    Raises ValueError if the script is outside `scripts/` or does not exist.
    """
    candidate = Path(describe_script)
    if not candidate.is_absolute():
        # Accept "describe_image.py" as well as "scripts/describe_image.py".
        candidate = _SCRIPTS_DIR / candidate.name
    candidate = candidate.resolve()
    if not candidate.is_relative_to(_SCRIPTS_DIR):
        raise ValueError(
            f"describe_script must be inside {_SCRIPTS_DIR}, got '{describe_script}'"
        )
    if not candidate.is_file():
        raise ValueError(f"describe_script '{describe_script}' not found")
    return candidate


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.
    
    Args:
        text (str): The string to check.
        
    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False
    
    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r'\$\$.*?\$\$',           # Block-level math formula $$...$$
        r'\$[^$\n]+?\$',          # Inline math formula $...$
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environment \begin{...}...\end{...}
        r'\\[a-zA-Z]+\{.*?\}',    # LaTeX command \command{...}
        r'\\[a-zA-Z]+',           # Simple LaTeX command \command
        r'\\\[.*?\\\]',           # Display math formula \[...\]
        r'\\\(.*?\\\)',           # Inline math formula \(...\)
    ]
    
    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.
    
    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r'\\documentclass\{[^}]+\}',  # \documentclass{...}
        r'\\usepackage\{[^}]+\}',    # \usepackage{...}
        r'\\usepackage\[[^\]]*\]\{[^}]+\}',  # \usepackage[options]{...}
        r'\\begin\{document\}',       # \begin{document}
        r'\\end\{document\}',         # \end{document}
    ]
    
    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text
    

def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.
    
    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if it's already enclosed in $$
    if text.startswith('$$') and text.endswith('$$'):
        text_new = text[2:-2].strip()
        if not '$' in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith('\\[') and text.endswith('\\]'):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"
        
    # Check if it's enclosed in \[ \]
    if len(re.findall(r'.*\\\[.*\\\].*', text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r'\$([^$]+)\$'
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text  

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):  
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if 'usepackage' in text:
        text = clean_latex_preamble(text)

    if text[0] == '`' and text[-1] == '`':
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text 


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.
    
    Args:
        text: The original text.
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple consecutive whitespace characters with a single space
    if text[:2] == '`$' and text[-2:] == '$`':
        text = text[1:-1]
    
    return text


def _describe_image_crop(image_crop: Image.Image, describe_script: str) -> str:
    """Run describe_script on a cropped image; return its description.

    Never raises: any failure (disallowed script, crash, timeout) is logged
    and returns "" so a single bad Picture cell cannot fail the whole page.
    """
    import subprocess
    import tempfile

    try:
        script = resolve_describe_script(describe_script)
    except ValueError as e:
        logger.warning("describe_script rejected: %s", e)
        return ""

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp_path = f.name
    try:
        image_crop.save(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(script), tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            logger.warning(
                "describe_script %s exited %d: %s",
                script, proc.returncode, proc.stderr.strip(),
            )
            return ""
        return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("describe_script %s failed: %s", script, e)
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def layoutjson2md(
    image: Image.Image,
    cells: list,
    text_key: str = 'text',
    no_page_hf: bool = False,
    image_mode: str = "base64",
    describe_script: str | None = None,
) -> str:
    """
    Converts a layout JSON format to Markdown.

    Args:
        image: A PIL Image object.
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_hf: If True, skips page headers and footers.
        image_mode: How to render Picture cells — "base64" (inline data URI),
            "file_ref" (plain filename tag, no file written), "describe"
            (call describe_script and embed its stdout as text), or "ocr"
            (emit the text the parser's picture OCR pass extracted, dropping
            the image itself).
        describe_script: Script used when image_mode="describe". Must live in
            the repository's scripts/ directory. Called as:
            python <describe_script> <image_path>; stdout is the description.

    Returns:
        str: The text in Markdown format.
    """
    text_items = []
    picture_idx = 0

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]
        text = cell.get(text_key, "")

        if no_page_hf and cell['category'] in ['Page-header', 'Page-footer']:
            continue

        if cell['category'] == 'Picture':
            if image_mode == "file_ref":
                text_items.append(f"![](picture_{picture_idx}.png)")
            elif image_mode == "ocr":
                # Text was filled in by the parser's picture OCR pass; skip the
                # cell entirely when the picture held no text.
                ocr_text = clean_text(text)
                if ocr_text:
                    text_items.append(ocr_text)
            elif image_mode == "describe" and describe_script:
                description = _describe_image_crop(
                    image.crop((x1, y1, x2, y2)), describe_script
                )
                if description:
                    text_items.append(f"> [Image: {description}]")
                else:
                    text_items.append("> [Image]")
            else:
                image_crop = image.crop((x1, y1, x2, y2))
                image_base64 = PILimage_to_base64(image_crop)
                text_items.append(f"![]({image_base64})")
            picture_idx += 1
        elif cell['category'] == 'Formula':
            text_items.append(get_formula_in_markdown(text))
        else:            
            text = clean_text(text)
            text_items.append(f"{text}")

    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def fix_streamlit_formulas(md: str) -> str:
    """
    Fixes the format of formulas in Markdown to ensure they display correctly in Streamlit.
    It adds a newline after the opening $$ and before the closing $$ if they don't already exist.
    
    Args:
        md_text (str): The Markdown text to fix.
        
    Returns:
        str: The fixed Markdown text.
    """
    
    # This inner function will be used by re.sub to perform the replacement
    def replace_formula(match):
        content = match.group(1)
        # If the content already has surrounding newlines, don't add more.
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        return f'$$\n{content}\n$$'
    
    # Use regex to find all $$....$$ patterns and replace them using the helper function.
    return re.sub(r'\$\$(.*?)\$\$', replace_formula, md, flags=re.DOTALL)
