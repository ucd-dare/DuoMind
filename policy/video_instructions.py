"""Shared instruction footer for recorded ManiSkill and RoboTwin RGB frames."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_HEIGHT = 288
MIN_WIDTH = 640


def video_size(width: int, height: int) -> tuple[int, int]:
    """Fixed, even encoder dimensions, independent of the current instructions."""
    output_width = max(MIN_WIDTH, width)
    output_width += output_width % 2
    image_height = round(height * output_width / width)
    image_height += image_height % 2
    return output_width, image_height + PANEL_HEIGHT


@lru_cache(maxsize=16)
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text, font, width):
    lines, line = [], ""
    # Measure pixels, including long words, instead of assuming a character width.
    for word in str(text).split():
        candidate = f"{line} {word}" if line else word
        if font.getlength(candidate) <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = ""
        for char in word:
            if line and font.getlength(line + char) > width:
                lines.append(line)
                line = ""
            line += char
    return lines + ([line] if line else [])


@lru_cache(maxsize=32)
def _panel(width, high_level, low_level):
    panel = Image.new("RGB", (width, PANEL_HEIGHT), (20, 25, 34))
    draw = ImageDraw.Draw(panel)
    rows = [
        ("HIGH-LEVEL TASK", high_level, (255, 220, 125)),
        ("LEFT ROBOT / LOW-LEVEL INSTRUCTION", low_level[0], (135, 205, 255)),
        ("RIGHT ROBOT / LOW-LEVEL INSTRUCTION", low_level[1], (150, 235, 175)),
    ]
    for index, (label, text, color) in enumerate(rows):
        top = index * 96
        draw.line((16, top, width - 16, top), fill=(55, 65, 80))
        draw.text((16, top + 7), label, font=_font(14), fill=color)
        text = text or "Awaiting instruction"
        for size in range(19, 7, -1):
            font = _font(size)
            lines = _wrap(text, font, width - 32)
            box = font.getbbox("Ag")
            line_height = box[3] - box[1] + 5
            if len(lines) * line_height <= 62:
                break
        else:
            raise ValueError("Instruction is too long for the video footer")
        for offset, line in enumerate(lines):
            draw.text((16, top + 28 + offset * line_height), line, font=font, fill=(240, 243, 248))
    return panel


def add_instruction_panel(frame, high_level, low_level):
    """Append captions outside the scene; callers supply the active VLA prompts."""
    if len(low_level) != 2:
        raise ValueError("Instruction videos require left and right robot prompts")
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    output_width, output_height = video_size(width, height)
    image_height = output_height - PANEL_HEIGHT
    scene = Image.fromarray(frame)
    if scene.size != (output_width, image_height):
        scene = scene.resize((output_width, image_height), Image.Resampling.BILINEAR)
    output = Image.new("RGB", (output_width, output_height))
    output.paste(scene, (0, 0))
    output.paste(_panel(output_width, str(high_level or ""), tuple(str(x or "") for x in low_level)), (0, image_height))
    return np.asarray(output)


def robotwin_video_frame(frame, model, task_instruction):
    """Read current planner commands, or the task prompt for direct policies."""
    agents = getattr(model, "agents", None)
    instructions = (
        [agent.low_level_instruction or task_instruction for agent in agents]
        if agents is not None
        else [task_instruction, task_instruction]
    )
    return add_instruction_panel(frame, task_instruction, instructions)
