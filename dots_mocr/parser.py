import os
import json
import threading
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool
import argparse
from PIL import Image

from dots_mocr.log import logger

from dots_mocr.model.inference import ModelOutputError, inference_with_vllm, is_backend_down, is_upstream_error
from dots_mocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_mocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_mocr.utils.doc_utils import fitz_doc_to_image, render_pdf_pages
from dots_mocr.utils.prompts import dict_promptmode_to_prompt
from dots_mocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes, parse_scene_text_output, post_process_scene_text, draw_scene_text_on_image, format_scene_text_to_markdown
from dots_mocr.utils.svg_utils import extract_svg_from_response, svg_to_png, create_comparison_image
from dots_mocr.utils.format_transformer import layoutjson2md


def page_error_result(page_no, exc) -> dict:
    """A per-page failure as a result entry the API layer turns into an ErrorItem."""
    code = "page_model_error" if is_upstream_error(exc) else "page_failed"
    logger.opt(exception=True).error(
        "page {} failed ({}): {}", page_no, code, exc
    )
    return {'page_no': page_no, 'error': {'code': code, 'message': str(exc)}}


class DotsMOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            protocol='http',
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=32768,
            num_thread=64,
            dpi = 200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
            fallback=None,
            fallback_cooldown=0.0,
        ):
        self.dpi = dpi

        # Optional second vLLM endpoint serving a same-format model, used when
        # the main one raises an upstream error. Keys: protocol, ip, port,
        # model_name, api_key. None disables it.
        self.fallback = fallback
        # After a main-model failure, send calls straight to the fallback for
        # this many seconds, so a wedged backend does not cost a full timeout
        # per page. 0 disables the breaker.
        self.fallback_cooldown = fallback_cooldown
        self._main_down_until = 0.0
        self._breaker_lock = threading.Lock()

        # default args for vllm server
        self.protocol = protocol
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        if self.use_hf:
            self._load_hf_model()
            logger.info("use hf model, num_thread will be set to 1")
        else:
            logger.info("use vllm model, num_thread will be set to {}", self.num_thread)
            if self.fallback:
                logger.info(
                    "fallback model configured: {}://{}:{} model={} cooldown={}s",
                    self.fallback.get("protocol", "http"), self.fallback.get("ip"),
                    self.fallback.get("port"), self.fallback.get("model_name"),
                    self.fallback_cooldown,
                )
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        from qwen_vl_utils import process_vision_info

        model_path = "./weights/DotsMOCR"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True,use_fast=True)
        self.process_vision_info = process_vision_info

    def _inference_with_hf(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def _inference_with_vllm(self, image, prompt, prompt_mode, temperature=None):
        """Run one inference, falling back to the secondary model if configured.

        Returns ``(content, used_fallback)``. Only upstream errors (backend down,
        timeout, 5xx, 429) trigger the fallback; anything else propagates. Only
        an unreachable backend arms the cooldown breaker.
        """
        system_prompt = "You are a helpful assistant."
        if prompt_mode != "prompt_general":
            system_prompt = None
        kwargs = dict(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
            system_prompt=system_prompt,
        )

        def call_main():
            return inference_with_vllm(
                image, prompt,
                model_name=self.model_name,
                protocol=self.protocol,
                ip=self.ip,
                port=self.port,
                **kwargs,
            )

        def call_fallback():
            fb = self.fallback
            return inference_with_vllm(
                image, prompt,
                model_name=fb.get("model_name", self.model_name),
                protocol=fb.get("protocol", self.protocol),
                ip=fb["ip"],
                port=fb.get("port", self.port),
                api_key=fb.get("api_key"),
                strip_reasoning=fb.get("strip_thinking", True),
                **kwargs,
            )

        if not self.fallback:
            return call_main(), False

        with self._breaker_lock:
            main_down = time.monotonic() < self._main_down_until
        if main_down:
            logger.debug("main model marked down; using fallback model")
            try:
                return call_fallback(), True
            except Exception as fb_exc:
                if not is_upstream_error(fb_exc):
                    raise
                # Both may not be down: main could have recovered before the
                # cooldown ran out, so probe it rather than fail the page.
                logger.warning(
                    "fallback model failed during main cooldown ({}); probing main", fb_exc
                )
                try:
                    content = call_main()
                except Exception as main_exc:
                    raise main_exc from fb_exc
                with self._breaker_lock:
                    self._main_down_until = 0.0
                logger.info("main model {} is back; breaker closed", self.model_name)
                return content, False

        try:
            return call_main(), False
        except Exception as main_exc:
            if not is_upstream_error(main_exc):
                raise
            logger.warning(
                "main model {} failed ({}); retrying on fallback model {}",
                self.model_name, main_exc, self.fallback.get("model_name", self.model_name),
            )
            if self.fallback_cooldown > 0 and is_backend_down(main_exc):
                with self._breaker_lock:
                    self._main_down_until = time.monotonic() + self.fallback_cooldown
            try:
                return call_fallback(), True
            except Exception as fb_exc:
                raise fb_exc from main_exc

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None, custom_prompt=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        if prompt_mode == 'prompt_image_to_svg':  # for SVG, pass the image size in as the viewbox
            prompt = prompt.replace("{width}", str(origin_image.width))
            prompt = prompt.replace("{height}", str(origin_image.height))
            logger.debug("svg prompt: {}", prompt)
        if prompt_mode == 'prompt_general':
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = "Please describe the content of this image."
        return prompt

    def _ocr_picture_cells(self, origin_image, cells):
        """Fill in `text` for Picture cells by OCR-ing each cropped region.

        The layout pass omits text for Picture cells by prompt instruction, so
        image_mode="ocr" recovers it with one extra inference per picture.

        Runs sequentially: pages already fan out across num_thread threads under
        the API's MOCR_MAX_CONCURRENT semaphore, so parallel crops here would
        multiply GPU concurrency again.

        Never raises: a bad bbox or a failed inference logs a warning and leaves
        the cell's text empty, so one picture cannot fail the whole page.

        Returns True when any crop was served by the fallback model.
        """
        prompt = dict_promptmode_to_prompt["prompt_ocr"]
        width, height = origin_image.width, origin_image.height
        used_fallback = False

        for cell in cells:
            if cell.get('category') != 'Picture':
                continue
            try:
                x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    logger.warning("skipping Picture cell with empty bbox: {}", cell.get('bbox'))
                    continue

                # Crops can fall below MIN_PIXELS; fetch_image upscales them to
                # a size the model accepts.
                crop = fetch_image(
                    origin_image.crop((x1, y1, x2, y2)),
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
                if self.use_hf:
                    response = self._inference_with_hf(crop, prompt)
                else:
                    response, crop_fallback = self._inference_with_vllm(crop, prompt, "prompt_ocr")
                    used_fallback = used_fallback or crop_fallback
                cell['text'] = (response or "").strip()
            except Exception as e:
                logger.warning("picture OCR failed for bbox {}: {}", cell.get('bbox'), e)
        return used_fallback

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        save_dir,
        save_name,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
        custom_prompt=None,
        temperature=None,
        image_mode="base64",
        describe_script=None,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels, custom_prompt=custom_prompt)

        logger.debug(
            "parse page: source={} page_idx={} origin={}x{} resized={}x{} "
            "min_pixels={} max_pixels={} prompt_mode={} prompt_len={}",
            source, page_idx, origin_image.width, origin_image.height,
            input_width, input_height, min_pixels, max_pixels, prompt_mode, len(prompt),
        )

        used_fallback = False
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response, used_fallback = self._inference_with_vllm(image, prompt, prompt_mode, temperature=temperature)
        if not response:
            logger.warning(
                "inference returned empty response: source={} page_idx={} prompt_mode={}",
                source, page_idx, prompt_mode,
            )
        else:
            logger.debug(
                "inference response: source={} page_idx={} len={} preview={!r}",
                source, page_idx, len(response), response[:120],
            )
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if not response:
            # Not an error: the page still yields (empty) output. Surfaced as a
            # diagnostic so a blank result is distinguishable from a blank page.
            result['empty_response'] = True
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr', 'prompt_web_parsing']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                bbox_scale=self.fallback.get("bbox_scale") if used_fallback else None,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                if response and response.strip() and not (isinstance(cells, str) and cells.strip()):
                    # The model answered, but none of it survived: a failed page,
                    # not a degraded one — "success" with empty text would hide it.
                    model = self.fallback.get("model_name", self.model_name) if used_fallback else self.model_name
                    raise ModelOutputError(
                        f"model {model} answered ({len(response)} chars) but it was not "
                        f"valid layout JSON and no text could be recovered"
                    )
                logger.debug(
                    "layout branch=filtered (json parse failed) page_idx={} md_chars={}",
                    page_idx, len(cells) if isinstance(cells, str) else 0,
                )
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                logger.debug(
                    "layout branch={} page_idx={} cells={}",
                    "layout_only" if prompt_mode == "prompt_layout_only_en" else "normal",
                    page_idx, len(cells) if isinstance(cells, list) else "n/a",
                )
                # Before the json dump and both layoutjson2md passes, so the
                # text reaches every output format and costs one call per
                # picture rather than two.
                if image_mode == "ocr" and prompt_mode != "prompt_layout_only_en":
                    used_fallback = self._ocr_picture_cells(origin_image, cells) or used_fallback

                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    logger.warning("Error drawing layout on image: {}", e)
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text', image_mode=image_mode, describe_script=describe_script)
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True, image_mode=image_mode, describe_script=describe_script)
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    logger.debug(
                        "markdown written: page_idx={} md_chars={} -> {}",
                        page_idx, len(md_content), md_file_path,
                    )
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        elif prompt_mode in ['prompt_scene_spotting']:
            instances, failed = post_process_scene_text(response, origin_image, image, min_pixels, max_pixels)
            
            # Draw visualization (fall back to the original image on failure).
            vis_image = origin_image if failed else draw_scene_text_on_image(origin_image, instances) if instances else origin_image
            
            # Save image
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            vis_image.save(image_layout_path)
            
            # Save JSON
            json_file_path = os.path.join(save_dir, f"{save_name}.json")
            with open(json_file_path, 'w', encoding="utf-8") as f:
                json.dump(instances if not failed else {"raw": response}, f, ensure_ascii=False, indent=2)
            
            # Save Markdown
            md_content = format_scene_text_to_markdown(instances) if not failed else response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            result.update({
                'layout_image_path': image_layout_path,
                'layout_info_path': json_file_path,
                'md_content_path': md_file_path,
                'text_instances': instances if not failed else None,
                'filtered': failed
            })

        elif prompt_mode in ['prompt_image_to_svg']:   ##todo
            svg_content, has_svg = extract_svg_from_response(response)
            
            if has_svg:
                # Convert SVG to PNG, preserving the original image aspect ratio.
                png_path = os.path.join(save_dir, f"{save_name}_rendered.png")
                w, h = origin_image.size
                tw, th = (1024, round(h * 1024 / w)) if w <= h else (round(w * 1024 / h), 1024)
                success, error = svg_to_png(svg_content, png_path, width=w, height=h)
                                
                if success:
                    # Build a comparison image: original on top, rendered below.
                    rendered_image = Image.open(png_path)
                    comparison_image = create_comparison_image(origin_image, rendered_image)
                    image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                    comparison_image.save(image_layout_path)
                else:
                    # SVG conversion failed, save the original image.
                    logger.warning("SVG to PNG failed: {}", error)
                    image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                    origin_image.save(image_layout_path)        
            else:
                # No SVG, save the original image.
                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
            
            # Markdown holds the raw model output directly.
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            md_content = f"# Generated SVG Code\n\n```xml\n{response}\n```"
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            result.update({
                'layout_image_path': image_layout_path,
                'md_content_path': md_file_path,
            })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        # Set last: picture OCR in the layout branch may also use the fallback.
        if used_fallback:
            result['fallback_model'] = self.fallback.get("model_name", self.model_name)
        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False, custom_prompt=None, temperature=None, image_mode="base64", describe_script=None):
        # Outside the try: an undecodable upload is the client's error, and the
        # API layer reports it as document_invalid rather than a failed page.
        origin_image = fetch_image(input_path)
        try:
            result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess, custom_prompt=custom_prompt, temperature=temperature, image_mode=image_mode, describe_script=describe_script)
        except Exception as exc:
            result = page_error_result(0, exc)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir, image_mode="base64", describe_script=None, start_page=0, end_page=None):
        logger.info("loading pdf: {} (prompt_mode={})", input_path, prompt_mode)
        pages, skipped = render_pdf_pages(
            input_path, dpi=self.dpi, start_page_id=start_page, end_page_id=end_page
        )
        total_pages = len(pages)
        if total_pages == 0 and not skipped:
            logger.warning("No renderable pages found in {}", input_path)
            return []
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source": "pdf",
                "page_idx": page_no,
                "image_mode": image_mode,
                "describe_script": describe_script,
            } for page_no, image in pages
        ]

        def _execute_task(task_args):
            # One bad page must not abort the pool and discard the pages that
            # already succeeded, so the failure travels back as a result entry.
            try:
                return self._parse_single_image(**task_args)
            except Exception as exc:
                return page_error_result(task_args["page_idx"], exc)

        if self.use_hf:
            num_thread =  1
        else:
            num_thread = min(total_pages, self.num_thread) if total_pages else 1
        logger.info(
            "Parsing PDF {} with {} pages using {} threads...",
            input_path, total_pages, num_thread,
        )

        start = time.monotonic()
        # Pages the renderer refused are reported too, at their real page number.
        results = [
            {'page_no': page_no, 'error': {'code': code, 'message': reason}}
            for page_no, code, reason in skipped
        ]
        if tasks:
            with ThreadPool(num_thread) as pool:
                with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                    for result in pool.imap_unordered(_execute_task, tasks):
                        if result.get("error"):
                            logger.debug(
                                "page failed: page_no={} code={}",
                                result.get("page_no"), result["error"].get("code"),
                            )
                        else:
                            logger.debug(
                                "page done: page_no={} has_md={} filtered={}",
                                result.get("page_no"),
                                bool(result.get("md_content_path")),
                                result.get("filtered", False),
                            )
                        results.append(result)
                        pbar.update(1)
        failed = sum(1 for r in results if r.get("error"))
        logger.info(
            "Parsed PDF {}: {}/{} pages ok, {} failed, in {:.2f}s",
            input_path, len(results) - failed, len(results), failed,
            time.monotonic() - start,
        )

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self, 
        input_path, 
        output_dir="", 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
        custom_prompt=None
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess, custom_prompt=custom_prompt)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        logger.info("Parsing finished, results saving to {}", save_dir)
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.mocr Multimodal OCR: Parse Anything from Documents",
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path"
    )
    
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--protocol", type=str, choices=['http', 'https'], default="http",
        help=""
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--use_hf", type=bool, default=False,
        help=""
    )
    parser.add_argument(
        "--custom_prompt", type=str, default=None,
        help="Custom prompt for free QA mode"
    )
    args = parser.parse_args()

    dots_mocr_parser = DotsMOCRParser(
        protocol=args.protocol,
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print(f"Using fitz preprocess for image input, check the change of the image pixels")
    result = dots_mocr_parser.parse_file(
        args.input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        fitz_preprocess=fitz_preprocess,
        )
    


if __name__ == "__main__":
    main()
