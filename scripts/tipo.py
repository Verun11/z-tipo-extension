import scripts

print(scripts, scripts.__file__, dir(scripts))

import os
import json
import pathlib
import random
from functools import lru_cache

import torch
import gradio as gr

import modules.scripts as scripts
import re # Needed for the new parser function
from modules import devices, shared, options
from modules.scripts import basedir, OnComponent
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
# from modules.prompt_parser import parse_prompt_attention # Keep for now, ensure calls are specific
from modules.extra_networks import parse_prompt
from modules.shared import opts

if hasattr(opts, "hypertile_enable_unet"):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None

import kgen.models as models
import kgen.executor.tipo as tipo
from kgen.executor.tipo import (
    parse_tipo_request,
    tipo_runner,
    apply_tipo_prompt,
    parse_tipo_result,
)
from kgen.formatter import seperate_tags, apply_format
from kgen.metainfo import TARGET, TIPO_DEFAULT_FORMAT
from kgen.logging import logger


ext_dir = basedir()
models.model_dir = pathlib.Path(ext_dir) / "models"


SEED_MAX = 2**31 - 1
QUOTESWAP = str.maketrans("'\"", "\"'")
TOTAL_TAG_LENGTH = {
    "VERY_SHORT": "very short",
    "SHORT": "short",
    "LONG": "long",
    "VERY_LONG": "very long",
}
PROCESSING_TIMING = {
    "BEFORE": "Before applying other prompt processings",
    "AFTER": "After applying other prompt processings",
}
DEFAULT_FORMAT = """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|general|>, 

<|quality|>, <|meta|>, <|rating|>"""
TIMING_INFO_TEMPLATE = (
    "_Prompt upsampling will be applied to {} "
    "sd-dynamic-promps and the webui's styles feature are applied_"
)
INFOTEXT_KEY = "TIPO Parameters"
INFOTEXT_KEY_PROMPT = "TIPO prompt"
INFOTEXT_NL_PROMPT = "TIPO nl prompt"
INFOTEXT_KEY_FORMAT = "TIPO format"

PROMPT_INDICATE_HTML = """
<div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center">
    <span>
        Original Prompt Loaded.<br>
        Click "Apply" to apply the original prompt.
    </span>
</div>
"""
RECOMMEND_MARKDOWN = """
### Recommended Model and Settings:

"""
MODEL_NAME_LIST = [
    f"{model_name} | {file}"
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


def on_process_timing_dropdown_changed(timing: str):
    info = ""
    if timing == PROCESSING_TIMING["BEFORE"]:
        info = "**only the first image in batch**, **before**"
    elif timing == PROCESSING_TIMING["AFTER"]:
        info = "**all images in batch**, **after**"
    else:
        raise ValueError(f"Unknown timing: {timing}")
    return TIMING_INFO_TEMPLATE.format(info)


def apply_strength(tag_map, strength_map, strength_map_nl, break_map):
    for cate in tag_map.keys():
        new_list = []
        # Skip natural language output at first
        if isinstance(tag_map[cate], str):
            # Ensure all the parts in the strength_map are in the prompt
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\(").replace(")", "\)")
                    part = part.replace("(", "\(").replace(")", "\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            tag_map[cate] = new_prompt
            continue
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\(").replace(")", "\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
            if tag in break_map or org_tag in break_map:
                new_list.append("BREAK")
        tag_map[cate] = new_list

    return tag_map


class TIPOScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None, None, None]
        self.tag_prompt_area = [None, None]
        self.prompt_area_row = [None, None]
        self.current_model = None
        self.on_after_component_elem_id = [
            ("txt2img_prompt_row", lambda x: self.create_new_prompt_area(0, x)),
            ("txt2img_prompt", lambda x: self.set_prompt_area(0, x)),
            ("img2img_prompt_row", lambda x: self.create_new_prompt_area(1, x)),
            ("img2img_prompt", lambda x: self.set_prompt_area(1, x)),
        ]

    def create_new_prompt_area(self, i2i: int, prompt_row: OnComponent):
        with prompt_row.component:
            with gr.Column(visible=not opts.tipo_no_extra_input):
                new_tag_prompt_area = gr.Textbox(
                    label="Tag Prompt",
                    lines=3,
                    placeholder="Tag Prompt for TIPO (Put Tags to Prompt region)",
                )
                new_prompt_area = gr.Textbox(
                    label="Natural Language Prompt",
                    lines=3,
                    placeholder="Natural Language Prompt for TIPO (Put Tags to Prompt region)",
                )
        self.tag_prompt_area[i2i] = new_tag_prompt_area
        self.prompt_area_row[i2i] = gr.Row()
        # with self.prompt_area_row[i2i]:
        self.prompt_area[i2i * 2 + 1] = new_prompt_area

    def set_prompt_area(self, i2i: int, component: OnComponent):
        self.prompt_area[i2i * 2] = component.component

    def title(self):
        return "TIPO"

    def show(self, _):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with self.prompt_area_row[is_img2img]:
            with gr.Column(
                scale=1, min_width=180, visible=not opts.tipo_no_extra_input
            ):
                prompt_gen = gr.Button(value="Generate Prompt")
            with gr.Column(scale=6):
                with (
                    InputAccordion(False, open=False, label=self.title())
                    if InputAccordion
                    else gr.Accordion(open=False, label=self.title())
                ) as tipo_acc:
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=1):
                                    if InputAccordion is None:
                                        enabled_check = gr.Checkbox(
                                            label="Enabled", value=False, min_width=20
                                        )
                                    else:
                                        enabled_check = tipo_acc
                                    read_orig_prompt_btn = gr.Button(
                                        size="sm",
                                        value="Apply original prompt",
                                        visible=False,
                                        min_width=20,
                                    )
                                with gr.Column(scale=3):
                                    orig_prompt_area = gr.TextArea(visible=False)
                                    orig_prompt_light = gr.HTML("")
                                orig_prompt_area.change(
                                    lambda x: PROMPT_INDICATE_HTML * bool(x),
                                    inputs=orig_prompt_area,
                                    outputs=orig_prompt_light,
                                )
                                orig_prompt_area.change(
                                    lambda x: gr.update(visible=bool(x)),
                                    inputs=orig_prompt_area,
                                    outputs=read_orig_prompt_btn,
                                )
                                read_orig_prompt_btn.click(
                                    fn=lambda x: x,
                                    inputs=[orig_prompt_area],
                                    outputs=self.prompt_area[is_img2img],
                                )

                            tag_length_choices = list(TOTAL_TAG_LENGTH.values()) + ["Custom"]
                            tag_length_radio = gr.Radio(
                                label="Tags Length target",
                                choices=tag_length_choices,
                                value=TOTAL_TAG_LENGTH["LONG"],
                            )
                            min_tags_slider = gr.Slider(
                                label="Min Custom Tag Amount",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=20,
                                visible=False
                            )
                            max_tags_slider = gr.Slider(
                                label="Max Custom Tag Amount",
                                minimum=1,
                                maximum=150,
                                step=1,
                                value=50,
                                visible=False
                            )
                            nl_length_radio = gr.Radio(
                                label="NL Length target",
                                choices=list(TOTAL_TAG_LENGTH.values()), # Assuming NL length doesn't need custom for now
                                value=TOTAL_TAG_LENGTH["LONG"],
                            )
                            ban_tags_textbox = gr.Textbox(
                                label="Ban tags",
                                info="Separate with comma. Regex supported.",
                                value="",
                                placeholder="umbrella, official.*, .*text, ...",
                            )
                            format_dropdown = gr.Dropdown(
                                label="Prompt Format",
                                info="The format you want to apply to final prompt",
                                choices=list(TIPO_DEFAULT_FORMAT.keys()) + ["custom"],
                                value="Both, tag first (recommend)",
                            )
                            format_textarea = gr.TextArea(
                                value=TIPO_DEFAULT_FORMAT[
                                    "Both, tag first (recommend)"
                                ],
                                label="Custom Prompt Format",
                                visible=False,
                                placeholder="<|extended|>. <|general|>",
                            )
                            format_dropdown.change(
                                lambda x: gr.update(
                                    visible=x == "custom",
                                    value=TIPO_DEFAULT_FORMAT.get(
                                        x, list(TIPO_DEFAULT_FORMAT.values())[0]
                                    ),
                                ),
                                inputs=format_dropdown,
                                outputs=format_textarea,
                            )

                            def on_tag_length_change(choice):
                                is_custom = choice == "Custom"
                                return gr.update(visible=is_custom), gr.update(visible=is_custom)

                            tag_length_radio.change(
                                fn=on_tag_length_change,
                                inputs=tag_length_radio,
                                outputs=[min_tags_slider, max_tags_slider]
                            )

                            with gr.Group():
                                with gr.Row():
                                    seed_num_input = gr.Number(
                                        label="Seed for upsampling tags",
                                        minimum=-1,
                                        maximum=2**31 - 1,
                                        step=1,
                                        scale=4,
                                        value=-1,
                                    )
                                    seed_random_btn = gr.Button(value="Randomize")
                                    seed_shuffle_btn = gr.Button(value="Shuffle")

                                    seed_random_btn.click(
                                        lambda: -1, outputs=[seed_num_input]
                                    )
                                    seed_shuffle_btn.click(
                                        lambda: random.randint(0, 2**31 - 1),
                                        outputs=[seed_num_input],
                                    )

                            with gr.Group():
                                process_timing_dropdown = gr.Dropdown(
                                    label="Upsampling timing",
                                    choices=list(PROCESSING_TIMING.values()),
                                    value=PROCESSING_TIMING["AFTER"],
                                )

                                process_timing_md = gr.Markdown(
                                    on_process_timing_dropdown_changed(
                                        process_timing_dropdown.value
                                    )
                                )

                                process_timing_dropdown.change(
                                    on_process_timing_dropdown_changed,
                                    inputs=[process_timing_dropdown],
                                    outputs=[process_timing_md],
                                )

                        with gr.Column():
                            gr.Markdown(RECOMMEND_MARKDOWN)
                            model_dropdown = gr.Dropdown(
                                label="Model",
                                choices=MODEL_NAME_LIST,
                                value=MODEL_NAME_LIST[0],
                            )
                            gguf_use_cpu = gr.Checkbox(label="Use CPU (GGUF)")
                            no_formatting = gr.Checkbox(
                                label="No formatting", value=False
                            )
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                info="← less random | more random →",
                                maximum=1.5,
                                minimum=0.1,
                                step=0.05,
                                value=0.5,
                            )
                            top_p_slider = gr.Slider(
                                label="Top-p",
                                info="← less unconfident tokens | more unconfident tokens →",
                                maximum=1,
                                minimum=0,
                                step=0.05,
                                value=0.95,
                            )
                            top_k_slider = gr.Slider(
                                label="Top-k",
                                info="← less unconfident tokens | more unconfident tokens →",
                                maximum=150,
                                minimum=0,
                                step=1,
                                value=80,
                            )
                            ignore_first_n_tags_slider = gr.Slider(
                                label="Ignore First N Tags",
                                minimum=0,
                                maximum=50,
                                step=1,
                                value=0,
                            )

        aspect_ratio_place_holder = gr.Number(value=1.0, visible=False)

        prompt_gen.click(
            self.prompt_gen_only,
            inputs=[
                self.tag_prompt_area[is_img2img],
                self.prompt_area[is_img2img * 2 + 1],
                aspect_ratio_place_holder,
                seed_num_input,
                tag_length_radio,
                min_tags_slider, # New
                max_tags_slider, # New
                nl_length_radio,
                ban_tags_textbox,
                format_dropdown,
                format_textarea,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                ignore_first_n_tags_slider,
                model_dropdown,
                gguf_use_cpu,
                no_formatting,
                self.tag_prompt_area[is_img2img],
            ],
            outputs=[
                self.prompt_area[is_img2img * 2],
            ],
        )

        self.infotext_fields = [
            (
                (tipo_acc, lambda d: gr.update(value=INFOTEXT_KEY in d))
                if InputAccordion
                else (tipo_acc, lambda d: gr.update(open=INFOTEXT_KEY in d))
            ),
            (
                self.prompt_area[is_img2img * 2],
                lambda d: d.get(INFOTEXT_KEY_PROMPT, d["Prompt"]),
            ),
            (
                self.prompt_area[is_img2img * 2 + 1],
                lambda d: d.get(INFOTEXT_NL_PROMPT, ""),
            ),
            (orig_prompt_area, lambda d: d["Prompt"]),
            (enabled_check, lambda d: INFOTEXT_KEY in d),
            (seed_num_input, lambda d: self.get_infotext(d, "seed", None)),
            (tag_length_radio, lambda d: self.get_infotext(d, "tag_length", TOTAL_TAG_LENGTH["LONG"])),
            (min_tags_slider, lambda d: self.get_infotext(d, "min_custom_tags", 20)),
            (max_tags_slider, lambda d: self.get_infotext(d, "max_custom_tags", 50)),
            (nl_length_radio, lambda d: self.get_infotext(d, "nl_length", None)), # Default for this one if not found
            (ban_tags_textbox, lambda d: self.get_infotext(d, "ban_tags", None)),
            (format_dropdown, lambda d: self.get_infotext(d, "format", None)), # Default for this one if not found
            (format_textarea, lambda d: d.get(INFOTEXT_KEY_FORMAT, None)),
            (
                process_timing_dropdown,
                lambda d: PROCESSING_TIMING.get(
                    self.get_infotext(d, "timing", None), None
                ),
            ),
            (temperature_slider, lambda d: self.get_infotext(d, "temperature", None)),
            (top_p_slider, lambda d: self.get_infotext(d, "top_p", None)),
            (top_k_slider, lambda d: self.get_infotext(d, "top_k", None)),
            (ignore_first_n_tags_slider, lambda d: self.get_infotext(d, "ignore_first_n_tags", 0)),
            (
                model_dropdown,
                lambda d: self.get_infotext(d, "model", None),
            ),
            (gguf_use_cpu, lambda d: self.get_infotext(d, "gguf_cpu", None)),
            (no_formatting, lambda d: self.get_infotext(d, "no_formatting", None)),
        ]

        return [
            enabled_check,
            process_timing_dropdown,
            seed_num_input,
            tag_length_radio,
            min_tags_slider, # New
            max_tags_slider, # New
            nl_length_radio,
            ban_tags_textbox,
            format_dropdown,
            format_textarea,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            ignore_first_n_tags_slider,
            model_dropdown,
            gguf_use_cpu,
            no_formatting,
            self.tag_prompt_area[is_img2img],
            self.prompt_area[is_img2img * 2 + 1],
        ]

    def get_infotext(self, d, target, default):
        return d.get(INFOTEXT_KEY, {}).get(target, default)

    def write_infotext(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        prompt: str,
        process_timing: str,
        seed: int,
        *args,
    ):
        p.extra_generation_params[INFOTEXT_KEY] = json.dumps(
            {
                "seed": seed,
                "timing": process_timing,
                "tag_length": args[0], # tag_length_radio
                "min_custom_tags": args[1], # min_tags_slider
                "max_custom_tags": args[2], # max_tags_slider
                "nl_length": args[3], # nl_length_radio
                "ban_tags": args[4], # ban_tags_textbox
                "format_selected": args[5], # format_dropdown
                "format": args[6], # format_textarea
                "temperature": args[7], # temperature_slider
                "top_p": args[8], # top_p_slider
                "top_k": args[9], # top_k_slider
                "ignore_first_n_tags": args[10], # ignore_first_n_tags_slider
                "model": args[11], # model_dropdown
                "gguf_cpu": args[12], # gguf_use_cpu
                "no_formatting": args[13], # no_formatting
            },
            ensure_ascii=False,
        ).translate(QUOTESWAP)
        p.extra_generation_params[INFOTEXT_KEY_PROMPT] = prompt.strip() or args[-1] # nl_prompt_area from UI
        p.extra_generation_params[INFOTEXT_NL_PROMPT] = args[-2] # tag_prompt_area from UI

        # If format_selected (args[5]) is "custom", save the custom format text (args[6])
        if args[5] == "custom":
            p.extra_generation_params[INFOTEXT_KEY_FORMAT] = args[6]

    def process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called after sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["AFTER"]:
            return

        self.original_prompt = p.all_prompts
        self.original_hr_prompt = getattr(p, "all_hr_prompts", None)
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(2**31 - 1)
        seed = int(seed)

        args = list(args)
        if args[3] != "custom":
            args[4] = TIPO_DEFAULT_FORMAT.get(args[3], args[4])

        self.write_infotext(p, p.prompt, "AFTER", seed, *args)

        args = list(args)
        nl_prompt = args.pop()
        new_all_prompts = []
        for prompt, sub_seed in zip(p.all_prompts, p.all_seeds):
            new_all_prompts.append(
                self._process(prompt, nl_prompt, aspect_ratio, seed + sub_seed, *args)
            )

        hr_fix_enabled = getattr(p, "enable_hr", False)

        if hr_fix_enabled:
            if p.hr_prompt != p.prompt:
                new_hr_prompts = []
                for prompt, hr_prompt in zip(p.all_prompts, p.all_hr_prompts):
                    if prompt == hr_prompt:
                        new_hr_prompts.append(prompt)
                    else:
                        new_hr_prompts.append(hr_prompt)
                p.all_hr_prompts = new_hr_prompts
            else:
                p.all_hr_prompts = new_all_prompts
        p.all_prompts = new_all_prompts

    def before_process(
        self,
        p: StableDiffusionProcessingTxt2Img | StableDiffusionProcessingImg2Img,
        is_enabled: bool,
        process_timing: str,
        seed: int,
        *args,
    ):
        """This method will be called before sd-dynamic-prompts and the styles are applied."""

        if not is_enabled:
            return

        if process_timing != PROCESSING_TIMING["BEFORE"]:
            return

        self.original_prompt = p.prompt
        self.original_hr_prompt = p.hr_prompt
        aspect_ratio = p.width / p.height
        if seed == -1:
            seed = random.randrange(4294967294)
        self.write_infotext(p, p.prompt, "BEFORE", seed, *args)
        seed = int(seed + p.seed)

        args = list(args)
        p.prompt = self._process(p.prompt, args.pop(), aspect_ratio, seed, *args)

    def prompt_gen_only(self, *args):
        args = list(args)
        seed = args[3]
        if seed == -1:
            seed = random.randrange(2**31 - 1)
            args[3] = seed
        return self._process(*args)

    def _process(
        self,
        prompt: str,
        nl_prompt: str,
        aspect_ratio: float,
        seed: int,
        tag_length: str, # tag_length_radio value e.g. "very short" or "Custom"
        min_tags: int,   # min_tags_slider value
        max_tags: int,   # max_tags_slider value
        nl_length: str,  # nl_length_radio
        ban_tags: str,   # ban_tags_textbox
        format_select: str, # format_dropdown
        format: str,     # format_textarea
        temperature: float, # temperature_slider
        top_p: float,    # top_p_slider
        top_k: int,      # top_k_slider
        ignore_first_n_tags: int, # ignore_first_n_tags_slider
        model: str,      # model_dropdown
        gguf_use_cpu: bool, # gguf_use_cpu
        no_formatting: bool, # no_formatting
        tag_prompt: str, # self.tag_prompt_area[is_img2img]
    ):
        normalize_tag = lambda t: t.strip().lower()
        BANNED_STRING_TOKENS = {"<|very_long|>", "<|long|>", "<|short|>", "<|very_short|>"}

        def is_numeric_only_string(s):
            if not isinstance(s, str):
                return False
            s = s.strip()
            if not s:
                return False
            try:
                float(s)
                return True
            except ValueError:
                return False

        prompt = prompt.strip() or tag_prompt
        seed = int(seed) % SEED_MAX
        if model != self.current_model:
            if " | " in model:
                model_name, gguf_name = model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = model
                gguf = False
            models.load_model(target, gguf, device="cpu" if gguf_use_cpu else "cuda")
            self.current_model = model
        prompt_preview = prompt.replace("\n", " ")[:40]
        logger.info(f"Processing prompt: {prompt_preview}...")
        logger.info(f"Processing with seed: {seed}")
        prompt_without_extranet, res = parse_prompt(prompt) # This is from extra_networks, not the attention parser
        logger.info(f"TIPO DEBUG: prompt_without_extranet received by TIPO script: '{prompt_without_extranet}'")

        # Use the local parse_prompt_attention for the main prompt
        prompt_parse_strength = _local_parse_prompt_attention(prompt_without_extranet, ignore_first_n_tags)

        # For NL prompt, also use local but with ignore_first_n_tags=0
        # Assuming modules.prompt_parser.parse_prompt_attention is still the one for nl_prompt if not using local.
        # To be safe and consistent, let's use the local one here too with 0.
        nl_prompt_parse_strength = _local_parse_prompt_attention(nl_prompt, 0)
        nl_prompt_processed_text = "" # Store the text part from nl_prompt_parse_strength
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt_processed_text += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))
        # The original nl_prompt is used later in parse_tipo_request, so we update it here
        # if it was modified by parse_attention (e.g. stripped of attention syntax)
        nl_prompt = nl_prompt_processed_text


        # Ensure the original nl_prompt (which might be just text without attention) is used below if needed
        # The variable `nl_prompt` is reassigned later by `parse_tipo_request`.
        # The `nl_prompt_parse_strength` is used to build `strength_map_nl`.
        # The text part of `nl_prompt` used in `parse_tipo_request` should be the one without attention syntax.
        # The current logic for nl_prompt seems to reconstruct it from parsed parts.

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))

        rebuild_extranet = ""
        for name, params in res.items():
            for param in params:
                items = ":".join(param.items)
                rebuild_extranet += f" <{name}:{items}>"

        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        break_map = set()
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            if part == "BREAK" and strength == -1:
                break_map.add(all_tags[-1])
                continue
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength

        # Determine effective_tag_length_target
        presets_mean_values = {
            "very short": 11.5,  # Approx mean for "very short" (6-17 tags)
            "short": 26.5,       # Approx mean for "short" (18-35 tags)
            "long": 44.5,        # Approx mean for "long" (36-53 tags)
            "very long": 63.5,   # Approx mean for "very long" (54-72 tags)
        }

        if tag_length == "Custom":
            custom_target_mean = (min_tags + max_tags) / 2
            # Find closest preset name based on mean values
            # Note: TOTAL_TAG_LENGTH uses "VERY_SHORT" as key, presets_mean_values uses "very short"
            # We need to map back to the keys kgen expects ("very_short", etc.)
            # Or, ensure preset keys here are what kgen expects if direct mapping.
            # The values from TOTAL_TAG_LENGTH are "very short", "short", etc.
            # kgen expects "very_short", "short", "long", "very_long".

            # Find the preset string (e.g., "very short") whose mean is closest
            closest_preset_str_value = min(presets_mean_values.keys(), key=lambda k: abs(presets_mean_values[k] - custom_target_mean))
            effective_tag_length_target = closest_preset_str_value # This will be like "very short"
        else:
            effective_tag_length_target = tag_length # This is already like "very short"

        # Convert to kgen expected format (e.g., "very_short")
        effective_tag_length_target = effective_tag_length_target.replace(" ", "_")

        # NL length processing (remains unchanged)
        processed_nl_length = nl_length.replace(" ", "_")

        org_tag_map = seperate_tags(all_tags)

        all_normalized_user_tags = set()
        if isinstance(org_tag_map, dict): # Ensure org_tag_map is a dict, as expected
            for category_tags_list in org_tag_map.values():
                if isinstance(category_tags_list, list): # Ensure tags are in a list
                    for tag in category_tags_list:
                        if isinstance(tag, str): # Ensure tag is a string
                            all_normalized_user_tags.add(normalize_tag(tag))

        # Store the initial nl_prompt (text part, after attention stripping) for reuse in iterations
        initial_nl_text_for_kgen = nl_prompt

        MAX_ITERATIONS = 3
        THRESHOLD_FOR_ITERATION = 72 # Max of "very_long" preset

        if tag_length == "Custom" and min_tags > THRESHOLD_FOR_ITERATION:
            logger.info(f"Starting iterative tag generation: min_tags={min_tags}, max_tags={max_tags}, target_preset={effective_tag_length_target}")
            accumulated_tags_set = set()
            final_addon_nl = ""
            base_seed = seed

            # This tag_map will be from the last iteration, used for formatted_prompt_by_tipo
            # and potentially for apply_strength if not building a new one.
            tag_map_from_last_run = {}

            for i in range(MAX_ITERATIONS):
                current_seed = base_seed + i
                logger.info(f"Iteration {i+1}/{MAX_ITERATIONS}, current_seed={current_seed}")

                # Use initial_nl_text_for_kgen for each call to parse_tipo_request
                meta, operations, general, current_iter_nl_prompt = parse_tipo_request(
                    org_tag_map, # Base user tags
                    initial_nl_text_for_kgen, # Original NL prompt text
                    tag_length_target=effective_tag_length_target,
                    nl_length_target=processed_nl_length, # Use consistent NL length target
                    generate_extra_nl_prompt=(not initial_nl_text_for_kgen and "<|extended|>" in format)
                                           or "<|generated|>" in format,
                )
                meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

                if isinstance(models.text_model, torch.nn.Module):
                    models.text_model.to(devices.device)

                current_tag_map, _ = tipo_runner(
                    meta,
                    operations,
                    general,
                    current_iter_nl_prompt, # NL prompt from current parse_tipo_request
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=current_seed,
                )
                tag_map_from_last_run = current_tag_map # Keep the latest for formatted output

                if isinstance(models.text_model, torch.nn.Module):
                    models.text_model.cpu()
                    devices.torch_gc()

                # Tag Extraction for this iteration
                temp_extended_nl_this_iter = ""
                temp_generated_nl_this_iter = ""

                for cate, tags_or_str in current_tag_map.items():
                    if isinstance(tags_or_str, str): # 'extended', 'generated'
                        if cate == "extended":
                            temp_extended_nl_this_iter = tags_or_str
                        elif cate == "generated":
                            temp_generated_nl_this_iter = tags_or_str
                    elif isinstance(tags_or_str, list):
                        # normalized_user_tags_for_category = {normalize_tag(t) for t in org_tag_map.get(cate, [])} # Removed
                        for tag_item in tags_or_str:
                            if normalize_tag(tag_item) not in all_normalized_user_tags: # Changed to global set
                                if tag_item not in BANNED_STRING_TOKENS and not is_numeric_only_string(tag_item):
                                    accumulated_tags_set.add(tag_item) # Add original tag_item

                # Decide which NL content to use from this iteration's findings
                # Store NL from the first iteration only, prioritize 'extended'
                if i == 0:
                    final_addon_nl = temp_extended_nl_this_iter or temp_generated_nl_this_iter

                logger.info(f"Iteration {i+1}: accumulated_tags_set size = {len(accumulated_tags_set)}")
                if len(accumulated_tags_set) >= min_tags:
                    break

            final_addon_tags_list = list(accumulated_tags_set)
            if len(final_addon_tags_list) > max_tags:
                logger.info(f"Truncating accumulated tags from {len(final_addon_tags_list)} to {max_tags}")
                final_addon_tags_list = final_addon_tags_list[:max_tags]

            addon = {"tags": final_addon_tags_list, "nl": final_addon_nl}
            tag_map = tag_map_from_last_run # Use the tag_map from the last iteration for apply_format

        else: # Single run logic (existing code, slightly restructured)
            logger.info(f"Starting single run tag generation: target={effective_tag_length_target}")
            meta, operations, general, current_nl_prompt_single_run = parse_tipo_request(
                org_tag_map,
                initial_nl_text_for_kgen, # Use initial NL text
                tag_length_target=effective_tag_length_target,
                nl_length_target=processed_nl_length,
                generate_extra_nl_prompt=(not initial_nl_text_for_kgen and "<|extended|>" in format)
                                       or "<|generated|>" in format,
            )
            meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            if isinstance(models.text_model, torch.nn.Module):
                models.text_model.to(devices.device)

            # This is the main tag_map for single run
            tag_map, _ = tipo_runner(
                meta,
                operations,
                general,
                current_nl_prompt_single_run, # NL prompt from this parse_tipo_request
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

            if isinstance(models.text_model, torch.nn.Module):
                models.text_model.cpu()
                devices.torch_gc()

            # Populate addon for single run
            addon_tags_single_run = []
            addon_nl_single_run = ""
            temp_nl_holder = {"extended": "", "generated": ""}

            for cate, tags_or_str in tag_map.items():
                if isinstance(tags_or_str, str):
                    if cate == "extended":
                        temp_nl_holder["extended"] = tags_or_str
                    elif cate == "generated":
                        temp_nl_holder["generated"] = tags_or_str
                elif isinstance(tags_or_str, list):
                    # normalized_user_tags_for_category = {normalize_tag(t) for t in org_tag_map.get(cate, [])} # Removed
                    for tag_item in tags_or_str:
                        if normalize_tag(tag_item) not in all_normalized_user_tags: # Changed to global set
                            if tag_item not in BANNED_STRING_TOKENS and not is_numeric_only_string(tag_item):
                                addon_tags_single_run.append(tag_item) # Add original tag_item

            addon_nl_single_run = temp_nl_holder["extended"] or temp_nl_holder["generated"]
            addon = {"tags": addon_tags_single_run, "nl": addon_nl_single_run}

        # Common downstream processing using the 'addon' dict and 'tag_map' (from last/only run)
        addon = apply_strength(addon, strength_map, strength_map_nl, break_map)
        unformatted_prompt_by_tipo = ( # This uses the (potentially large) accumulated list of tags
            prompt + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"]
        )

        # This apply_strength uses the tag_map from the last/only run.
        # For iterative mode, this means it won't reflect all accumulated tags if they were from different categories.
        # This is a known simplification as per subtask description.
        tag_map = apply_strength(tag_map, strength_map, strength_map_nl, break_map)
        formatted_prompt_by_tipo = apply_format(tag_map, format).replace(
            "BREAK,", "BREAK"
        )

        if no_formatting:
            final_prompt = unformatted_prompt_by_tipo
        else:
            final_prompt = formatted_prompt_by_tipo

        result = final_prompt + "\n" + rebuild_extranet
        logger.info("Prompt processing done.")
        return result


def parse_infotext(_, params):
    try:
        params[INFOTEXT_KEY] = json.loads(params[INFOTEXT_KEY].translate(QUOTESWAP))
    except Exception:
        pass


# Definition of the local parser function, copied and adapted from nodes/tipo.py
# Added re import at the top of the file.
# Added _local_ prefix to avoid potential conflicts if original import is still used elsewhere.
_re_attention_scripts = re.compile(
    r"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|\]|[^\\()\[\]:]+|:",
    re.X,
)
_re_break_scripts = re.compile(r"\s*\bBREAK\b\s*", re.S)

def _local_parse_prompt_attention(text, ignore_first_n_tags: int = 0):
    if ignore_first_n_tags > 0:
        if "," in text:
            tags_list = text.split(",", ignore_first_n_tags)
            if len(tags_list) > ignore_first_n_tags:
                text = tags_list[-1].lstrip()
            else:
                text = ""
        elif not text.strip():
            text = ""
        else:
            text = ""

    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in _re_attention_scripts.finditer(text):
        text_segment = m.group(0)
        weight = m.group(1)

        if text_segment.startswith("\\"):
            res.append([text_segment[1:], 1.0])
        elif text_segment == "(":
            round_brackets.append(len(res))
        elif text_segment == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets: # Check round_brackets for safety, though finditer implies structure
            multiply_range(round_brackets.pop(), float(weight))
        elif text_segment == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_segment == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(_re_break_scripts, text_segment)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part: # Ensure part is not empty before appending
                    res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res: # Changed from len(res) == 0 to not res for conciseness
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0] is not None and res[i+1][0] is not None : # Ensure text parts are not None
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res

scripts.script_callbacks.on_infotext_pasted(parse_infotext)

options.categories.register_category("prompt_gen", "Prompt Gen")
shared.options_templates.update(
    shared.options_section(
        ("TIPO", "TIPO", "prompt_gen"),
        {
            "tipo_no_extra_input": shared.OptionInfo(
                False,
                (
                    "Disable extra input for TIPO"
                    ", Natural Language Prompt and Tag Prompt will be hidden."
                    " (UI Reload Needed)"
                ),
            ),
        },
    )
)
