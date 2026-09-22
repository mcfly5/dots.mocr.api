dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_ocr: parse ocr text except the Page-header and Page-footer
    "prompt_ocr": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # prompt_web_parsing: parse all webpage layout info in json format.
    "prompt_web_parsing": """Parsing the layout info of this webpage image with format json:\n""",

    # prompt_scene_spotting: scene spotting
    "prompt_scene_spotting": """Detect and recognize the text in the image.""",
    
    # prompt_img2svg: generate the SVG code of the image
    "prompt_image_to_svg": """Please generate the SVG code based on the image.viewBox="0 0 {width} {height}\"""",

    # prompt_free_qa: general prompt 
    "prompt_general": """ """,

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}


# Prompts for a general-purpose VLM serving as the fallback model (e.g. Qwen-VL),
# which does not know dots.mocr's prompt conventions. Answers must still
# post-process like dots output: normalize_layout_response() maps bbox_2d to
# bbox and strips code fences, and VLLM_FALLBACK_BBOX_SCALE handles relative
# coordinates. Modes missing here send the fallback the dots prompt.
_LAYOUT_CATEGORIES = "Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title"

dict_promptmode_to_fallback_prompt = {
    "prompt_layout_all_en": f"""Detect every layout element in this document image and read its text.

Output a JSON array with one object per element, in human reading order.

- category: exactly one of {_LAYOUT_CATEGORIES}.
- text: the original text of the element, not translated.
  - Table: always an HTML <table>, one <tr> per row and one <td> per cell. Never a Markdown table or plain text.
  - Formula: LaTeX.
  - Picture: omit the "text" key.
  - Everything else: Markdown.
- Output only the JSON array, with no explanation.

Example:
[
  {{"bbox_2d": [80, 40, 520, 70], "category": "Section-header", "text": "## Results"}},
  {{"bbox_2d": [80, 80, 900, 130], "category": "Text", "text": "Samples were measured twice."}},
  {{"bbox_2d": [80, 140, 900, 260], "category": "Table", "text": "<table><tr><td>Sample</td><td>Fat, %</td></tr><tr><td>A</td><td>3.2</td></tr><tr><td>B</td><td>2.5</td></tr></table>"}},
  {{"bbox_2d": [80, 270, 600, 300], "category": "Formula", "text": "\\\\bar{{x}} = \\\\frac{{1}}{{n}}\\\\sum_{{i=1}}^{{n}} x_i"}}
]
""",

    "prompt_layout_only_en": f"""Detect every layout element in this document image.

Output a JSON array with one object per element, in human reading order:
[{{"bbox_2d": [x1, y1, x2, y2], "category": "..."}}, ...]

- category: exactly one of {_LAYOUT_CATEGORIES}.
- Do not output the text of the elements.
- Output only the JSON array, with no explanation.
""",

    "prompt_ocr": """Extract all text from this image in reading order, without translating it. Format tables as HTML <table> (never Markdown tables or plain text), formulas as LaTeX and everything else as Markdown. Output only the extracted text, with no explanation.""",
}
