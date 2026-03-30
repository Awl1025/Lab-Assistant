"""
AI agent for lab image analysis.

Cleaned version:
- consistent lesion report format
- direct routing for lesion requests and direct image paths
- robust handling of quoted paths with spaces
- simpler agent fallback that only returns the final message
- reduced conversational junk in output
- suppresses noisy library logging from google_genai/httpx/urllib3
"""

import os
import base64
import json
import re
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv

import numpy as np
from PIL import Image, ImageDraw

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

load_dotenv()

# Quiet logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

for noisy_logger in ["google_genai", "httpx", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)


# =========================
# Config
# =========================
CHROMA_DIR = Path("./chroma_db")
EMBED_MODEL = "models/text-embedding-004"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K_PAPERS = 5
IMAGE_EXT_RE = r"(?:png|jpg|jpeg|tif|tiff|bmp|gif)"

SYSTEM_PROMPT = """You are an expert biomedical image analysis assistant with deep knowledge of
cell biology, fluorescence microscopy, cardiac ablation procedures, and bioimage analysis.

You have access to these tools:
- describe_microscopy_image
- extract_mask_stats
- search_papers_for_image
- analyze_ablation_lesions
- convert_pixels_to_mm

Rules:
- For ablation/cardiac lesion images, always use analyze_ablation_lesions first.
- Do not rewrite, summarize, or reformat the lesion analysis tool output.
- Return the lesion analysis tool output exactly as produced whenever it is used.
- Measurements must always clearly state whether they are in mm or pixels.
- If no ruler is detected, report pixels and say that no reliable ruler or scale bar was found.
- For non-lesion image analysis, answer directly and concisely.
- Do not include conversational filler.
- Do not ask unnecessary follow-up questions.
- If the request can be completed, do it.
"""


# =========================
# Data models
# =========================
@dataclass
class ScaleInfo:
    unit: str = "px"
    mm_per_pixel: Optional[float] = None
    source: str = "none"
    confidence: str = "unknown"
    note: str = "No reliable ruler or scale bar was detected. Measurements are reported in pixels."


@dataclass
class RulerDetection:
    ruler_found: bool
    ruler_span_fraction: float
    ruler_length_mm: float
    ruler_location: str
    confidence: str


@dataclass
class Lesion:
    id: int
    description: str
    bbox_relative: list[float]
    width_px: float
    depth_px: float
    shape: str
    confidence: str

    def estimated_area_px2(self) -> float:
        return float(np.pi * (self.width_px / 2.0) * (self.depth_px / 2.0))


@dataclass
class LesionAnalysis:
    lesion_count: int
    lesions: list[Lesion] = field(default_factory=list)
    tissue_notes: str = "N/A"


# =========================
# Helpers
# =========================
def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def prep_image_for_gemini(path: Path) -> str:
    """Convert image to base64 PNG for Gemini vision. Returns base64 string."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        img = Image.open(str(path))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        img.save(tmp_path, format="PNG")
        return encode_image_base64(tmp_path)
    return encode_image_base64(str(path))


def load_mask_array(path: str) -> np.ndarray:
    try:
        import tifffile as tiff
        if path.lower().endswith((".tif", ".tiff")):
            arr = tiff.imread(path).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return (arr > 0).astype(np.uint8)
    except ImportError:
        logger.warning("tifffile not installed; falling back to PIL.")

    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def get_vectorstore():
    if not CHROMA_DIR.exists():
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    except Exception as e:
        logger.exception("Failed to initialize vector store.")
        raise RuntimeError(f"Failed to initialize vector store: {e}") from e


def ask_gemini_vision(b64: str, prompt: str) -> str:
    """Send an image + prompt to Gemini and return text response."""
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
        resp = llm.invoke([message])
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        raise RuntimeError(f"Gemini vision request failed: {e}") from e


def strip_code_fences(text: str) -> str:
    return re.sub(r"```json|```", "", text).strip()


def safe_json_load(raw: str) -> dict[str, Any]:
    cleaned = strip_code_fences(raw)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{raw}")
    return json.loads(match.group(0))


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_bbox_relative(bbox: Any) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    vals = [max(0.0, min(1.0, coerce_float(v))) for v in bbox]
    x1, y1, x2, y2 = vals

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def parse_ruler_detection(raw: str, img_width_px: int) -> ScaleInfo:
    try:
        data = safe_json_load(raw)
        detection = RulerDetection(
            ruler_found=bool(data.get("ruler_found", False)),
            ruler_span_fraction=coerce_float(data.get("ruler_span_fraction"), 0.0),
            ruler_length_mm=coerce_float(data.get("ruler_length_mm"), 0.0),
            ruler_location=str(data.get("ruler_location", "unknown")),
            confidence=str(data.get("confidence", "unknown")),
        )

        if detection.ruler_found and detection.ruler_span_fraction > 0 and detection.ruler_length_mm > 0:
            ruler_px = detection.ruler_span_fraction * img_width_px
            mm_per_pixel = detection.ruler_length_mm / ruler_px if ruler_px > 0 else None

            if mm_per_pixel and mm_per_pixel > 0:
                return ScaleInfo(
                    unit="mm",
                    mm_per_pixel=mm_per_pixel,
                    source="detected_ruler",
                    confidence=detection.confidence,
                    note=(
                        f"Ruler detected ({detection.confidence} confidence): "
                        f"{detection.ruler_length_mm:.2f} mm spans about {ruler_px:.0f} px "
                        f"at {detection.ruler_location}, giving {mm_per_pixel:.6f} mm/pixel."
                    ),
                )
    except Exception as e:
        logger.warning("Ruler parse failed: %s", e)

    return ScaleInfo()


def parse_lesion_analysis(raw: str) -> LesionAnalysis:
    data = safe_json_load(raw)
    raw_lesions = data.get("lesions", [])

    lesions: list[Lesion] = []
    for idx, item in enumerate(raw_lesions, start=1):
        lesions.append(
            Lesion(
                id=coerce_int(item.get("id"), idx),
                description=str(item.get("description", "")).strip(),
                bbox_relative=validate_bbox_relative(item.get("bbox_relative")),
                width_px=coerce_float(item.get("width_px")),
                depth_px=coerce_float(item.get("depth_px")),
                shape=str(item.get("shape", "unknown")).strip(),
                confidence=str(item.get("confidence", "unknown")).strip(),
            )
        )

    lesion_count = coerce_int(data.get("lesion_count"), len(lesions))
    tissue_notes = str(data.get("tissue_notes", "N/A")).strip()

    if lesion_count != len(lesions):
        lesion_count = len(lesions)

    return LesionAnalysis(
        lesion_count=lesion_count,
        lesions=lesions,
        tissue_notes=tissue_notes,
    )


def detect_ruler(b64: str, img_width_px: int) -> ScaleInfo:
    ruler_prompt = """Examine this cardiac ablation gross pathology image carefully.

Is there a ruler, scale bar, or measuring device visible in the image?

Return valid JSON only:
{
  "ruler_found": true,
  "ruler_span_fraction": 0.25,
  "ruler_length_mm": 10,
  "ruler_location": "bottom-right",
  "confidence": "high"
}

If no ruler is visible, return:
{
  "ruler_found": false,
  "ruler_span_fraction": 0.0,
  "ruler_length_mm": 0,
  "ruler_location": "unknown",
  "confidence": "low"
}
"""
    raw = ask_gemini_vision(b64, ruler_prompt)
    return parse_ruler_detection(raw, img_width_px)


def detect_lesions(b64: str, img_w: int, img_h: int) -> LesionAnalysis:
    lesion_prompt = f"""This is a cardiac ablation gross pathology image (cross-section of heart tissue).
The image is {img_w} x {img_h} pixels.

Identify all visible ablation lesions. Ablation lesions often appear as dark brown or tan-brown
regions in myocardium and may have a pale center with darker border.

Return valid JSON only in this exact format:
{{
  "lesion_count": 2,
  "lesions": [
    {{
      "id": 1,
      "description": "dark brown oval lesion in upper-left myocardium",
      "bbox_relative": [0.10, 0.15, 0.22, 0.30],
      "width_px": 120,
      "depth_px": 85,
      "shape": "oval",
      "confidence": "high"
    }}
  ],
  "tissue_notes": "General observations about what the image shows"
}}

Rules:
- bbox_relative must be fractions from 0.0 to 1.0
- width_px and depth_px must be numeric
- describe each lesion briefly and specifically
- include tissue_notes as a concise summary of the overall image
- if uncertain, use best estimate
"""
    raw = ask_gemini_vision(b64, lesion_prompt)
    return parse_lesion_analysis(raw)


def draw_lesion_annotations(image_path: Path, lesions: list[Lesion], scale_info: ScaleInfo) -> str:
    img = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = [
        "#FF4444", "#FF8800", "#FFDD00", "#44FF44", "#4488FF",
        "#AA44FF", "#FF44AA", "#00FFFF", "#FF6666", "#88FF88"
    ]

    for i, lesion in enumerate(lesions):
        color = colors[i % len(colors)]
        x1f, y1f, x2f, y2f = lesion.bbox_relative

        x1 = int(x1f * w)
        y1 = int(y1f * h)
        x2 = int(x2f * w)
        y2 = int(y2f * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if scale_info.unit == "mm" and scale_info.mm_per_pixel:
            width_val = lesion.width_px * scale_info.mm_per_pixel
            depth_val = lesion.depth_px * scale_info.mm_per_pixel
            label = f"L{lesion.id} W:{width_val:.2f} D:{depth_val:.2f} mm"
        else:
            label = f"L{lesion.id} W:{lesion.width_px:.0f} D:{lesion.depth_px:.0f} px"

        label_w = max(110, len(label) * 7 + 8)
        label_top = max(0, y1 - 18)
        draw.rectangle([x1, label_top, x1 + label_w, y1], fill=color)
        draw.text((x1 + 3, label_top + 2), label, fill="black")

    out_path = Path(tempfile.gettempdir()) / f"annotated_{image_path.stem}.png"
    img.save(out_path)
    return str(out_path)


def format_lesion_report(
    image_path: Path,
    img_w: int,
    img_h: int,
    scale_info: ScaleInfo,
    analysis: LesionAnalysis,
    annotated_path: Optional[str] = None,
) -> str:
    unit = scale_info.unit
    mpp = scale_info.mm_per_pixel

    lines: list[str] = []
    total_area = 0.0

    lines.append(f"I found {analysis.lesion_count} cardiac ablation lesion(s) in the image.")
    lines.append("")
    lines.append("Lesions:")

    if not analysis.lesions:
        lines.append("- No definite lesions were detected.")
        lines.append("")
    else:
        for lesion in analysis.lesions:
            area_px2 = lesion.estimated_area_px2()

            lines.append(f"- Lesion {lesion.id}: {lesion.description or 'No description provided'}")
            lines.append(f"  Shape: {lesion.shape}")
            lines.append(f"  Confidence: {lesion.confidence}")

            if unit == "mm" and mpp:
                width_val = lesion.width_px * mpp
                depth_val = lesion.depth_px * mpp
                area_val = area_px2 * (mpp ** 2)
                total_area += area_val

                lines.append(f"  Width: {width_val:.2f} mm")
                lines.append(f"  Depth: {depth_val:.2f} mm")
                lines.append(f"  Estimated area: {area_val:.2f} mm²")
            else:
                total_area += area_px2
                lines.append(f"  Width: {lesion.width_px:.0f} px")
                lines.append(f"  Depth: {lesion.depth_px:.0f} px")
                lines.append(f"  Estimated area: {area_px2:.0f} px²")

            lines.append("")

    if unit == "mm":
        lines.append(f"Total estimated lesion area: {total_area:.2f} mm².")
    else:
        lines.append(f"Total estimated lesion area: {total_area:.0f} px².")
        lines.append("All measurements are reported in pixels because no reliable ruler or scale bar was detected.")
    lines.append("")

    lines.append("Image writeup:")
    if analysis.tissue_notes and analysis.tissue_notes != "N/A":
        lines.append(analysis.tissue_notes)
    else:
        lines.append(
            "The image shows cardiac tissue with visible ablation lesions. "
            "The lesions appear as darker, well-demarcated regions compared with the surrounding myocardium."
        )

    lines.append("")
    lines.append("Image details:")
    lines.append(f"- File: {image_path.name}")
    lines.append(f"- Image size: {img_w} x {img_h} px")
    lines.append(f"- Scale source: {scale_info.source}")
    lines.append(f"- Scale confidence: {scale_info.confidence}")
    lines.append(f"- Scale note: {scale_info.note}")

    if annotated_path:
        lines.append(f"- Annotated image: {annotated_path}")

    return "\n".join(lines)


def is_supported_image_path(path_str: str) -> bool:
    return bool(re.search(rf"\.({IMAGE_EXT_RE})$", path_str, re.IGNORECASE))


def clean_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        return text[1:-1].strip()
    return text


def extract_file_path_from_text(text: str) -> Optional[str]:
    text = text.strip()

    cleaned = clean_wrapping_quotes(text)
    if is_supported_image_path(cleaned):
        return cleaned

    double_quoted = re.search(
        rf'"([^"]+\.{IMAGE_EXT_RE})"',
        text,
        re.IGNORECASE
    )
    if double_quoted:
        return double_quoted.group(1)

    single_quoted = re.search(
        rf"'([^']+\.{IMAGE_EXT_RE})'",
        text,
        re.IGNORECASE
    )
    if single_quoted:
        return single_quoted.group(1)

    plain_match = re.search(
        rf'([^\s\'"]+\.{IMAGE_EXT_RE})',
        text,
        re.IGNORECASE
    )
    if plain_match:
        return plain_match.group(1)

    return None


def extract_scale_from_text(text: str) -> Optional[float]:
    scale_match = re.search(
        r'(\d*\.?\d+)\s*(?:mm/pixel|mm per pixel|mm_per_pixel)',
        text,
        re.IGNORECASE
    )
    if scale_match:
        try:
            return float(scale_match.group(1))
        except ValueError:
            return None
    return None


def parse_lesions_command_args(raw_args: str) -> tuple[str, Optional[float]]:
    raw_args = raw_args.strip()
    if not raw_args:
        return "", None

    quoted_match = re.match(
        rf"""^\s*(['"])(.+?\.{IMAGE_EXT_RE})\1(?:\s+(\d*\.?\d+))?\s*$""",
        raw_args,
        re.IGNORECASE
    )
    if quoted_match:
        path = quoted_match.group(2).strip()
        scale = quoted_match.group(3)
        return path, float(scale) if scale else None

    scale_match = re.match(
        rf"""^(.*\.{IMAGE_EXT_RE})\s+(\d*\.?\d+)\s*$""",
        raw_args,
        re.IGNORECASE
    )
    if scale_match:
        path = scale_match.group(1).strip()
        scale = float(scale_match.group(2))
        return path, scale

    return raw_args, None


def is_lesion_related_request(text: str) -> bool:
    lower = text.lower()
    keywords = [
        "lesion", "lesions", "ablation", "cardiac ablation",
        "myocardium", "heart tissue", "gross pathology"
    ]
    return any(k in lower for k in keywords)


# =========================
# Tools
# =========================
@tool
def analyze_ablation_lesions(image_path: str, mm_per_pixel: float = 0.0) -> str:
    """
    Detect and measure cardiac ablation lesions in a gross pathology image.
    Identifies lesion count, width, depth, and estimated area per lesion.
    Automatically detects a ruler for mm scaling; falls back to pixels.
    Use mm_per_pixel to manually override scale.
    """
    path = Path(image_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    try:
        img = Image.open(str(path))
        img_w, img_h = img.size
    except Exception as e:
        return f"Error: Could not open image '{path}': {e}"

    try:
        b64 = prep_image_for_gemini(path)
    except Exception as e:
        return f"Error: Failed to prepare image for Gemini: {e}"

    try:
        scale_info = detect_ruler(b64, img_w)
    except Exception as e:
        logger.warning("Ruler detection failed: %s", e)
        scale_info = ScaleInfo(note=f"Ruler detection failed, so measurements are reported in pixels. Details: {e}")

    if mm_per_pixel > 0:
        scale_info = ScaleInfo(
            unit="mm",
            mm_per_pixel=mm_per_pixel,
            source="manual_override",
            confidence="user-supplied",
            note=f"Manual scale applied: {mm_per_pixel:.6f} mm/pixel.",
        )

    try:
        analysis = detect_lesions(b64, img_w, img_h)
    except Exception as e:
        return f"Error: Lesion detection failed: {e}"

    annotated_path = None
    try:
        annotated_path = draw_lesion_annotations(path, analysis.lesions, scale_info)
    except Exception as e:
        logger.warning("Annotation failed: %s", e)

    return format_lesion_report(
        image_path=path,
        img_w=img_w,
        img_h=img_h,
        scale_info=scale_info,
        analysis=analysis,
        annotated_path=annotated_path,
    )


@tool
def convert_pixels_to_mm(pixels: float, mm_per_pixel: float) -> str:
    """
    Convert a pixel measurement to millimeters using a known scale ratio.
    """
    try:
        pixels = float(pixels)
        mm_per_pixel = float(mm_per_pixel)
    except Exception:
        return "Error: pixels and mm_per_pixel must be numeric."

    mm = pixels * mm_per_pixel
    area_mm2 = np.pi * (pixels / 2) ** 2 * (mm_per_pixel ** 2)

    return (
        f"Conversion using scale: {mm_per_pixel:.6f} mm/pixel\n"
        f"Length: {pixels:.1f} px -> {mm:.3f} mm\n"
        f"If circular: area = {area_mm2:.3f} mm²"
    )


@tool
def describe_microscopy_image(image_path: str) -> str:
    """
    Analyze any lab/microscopy image using Gemini vision.
    Describe morphology, density, staining patterns, and notable features.
    """
    path = Path(image_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    try:
        b64 = prep_image_for_gemini(path)
        prompt = (
            "You are an expert bioimage analyst. Analyze this lab image and provide:\n"
            "1. Tissue/cell morphology\n"
            "2. Approximate density or distribution of structures\n"
            "3. Staining or visual pattern observations\n"
            "4. Any abnormalities or notable features\n"
            "5. Suggested analysis approach\n"
            "Be concise and scientifically precise."
        )
        return ask_gemini_vision(b64, prompt)
    except Exception as e:
        return f"Error: microscopy analysis failed: {e}"


@tool
def extract_mask_stats(mask_path: str) -> str:
    """
    Extract quantitative statistics from a binary segmentation mask.
    Returns object count, foreground coverage %, and size stats.
    """
    from scipy import ndimage as ndi

    path = Path(mask_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    try:
        mask = load_mask_array(str(path))
    except Exception as e:
        return f"Error: failed to load mask '{path}': {e}"

    total_pixels = mask.size
    foreground_pixels = int(mask.sum())
    coverage_pct = round(100.0 * foreground_pixels / total_pixels, 2)

    labeled, num_objects = ndi.label(mask)
    sizes = np.array(ndi.sum(mask, labeled, range(1, num_objects + 1)))

    mean_size = round(float(sizes.mean()), 1) if len(sizes) > 0 else 0.0
    median_size = round(float(np.median(sizes)), 1) if len(sizes) > 0 else 0.0
    min_size = int(sizes.min()) if len(sizes) > 0 else 0
    max_size = int(sizes.max()) if len(sizes) > 0 else 0
    std_size = round(float(sizes.std()), 1) if len(sizes) > 0 else 0.0

    return (
        f"Mask Statistics: {path.name}\n"
        f"Image size: {mask.shape[1]} x {mask.shape[0]} px\n"
        f"Detected objects: {num_objects}\n"
        f"Foreground: {foreground_pixels} px ({coverage_pct}%)\n"
        f"Object sizes: mean={mean_size} | median={median_size} | "
        f"std={std_size} | min={min_size} | max={max_size} px"
    )


@tool
def search_papers_for_image(query: str) -> str:
    """
    Search the scientific paper vector store for content related to a query.
    """
    try:
        db = get_vectorstore()
    except Exception as e:
        return f"Error: vector store unavailable: {e}"

    if db is None:
        return "No vector store found at './chroma_db'. Run the paper indexing step first."

    try:
        results = db.similarity_search(query, k=TOP_K_PAPERS)
    except Exception as e:
        return f"Error: similarity search failed: {e}"

    if not results:
        return f"No relevant papers found for query: '{query}'."

    lines = [f"Top {len(results)} paper excerpts for: '{query}'", ""]
    for i, doc in enumerate(results, start=1):
        title = doc.metadata.get("title", "Unknown")
        dataset = doc.metadata.get("dataset", "Unknown")
        paper_id = doc.metadata.get("paper_id") or doc.metadata.get("pmid", "?")
        snippet = doc.page_content[:400].replace("\n", " ")
        lines.append(f"{i}. [{dataset}] {title} (ID: {paper_id})")
        lines.append(f"   ...{snippet}...")
        lines.append("")
    return "\n".join(lines)


# =========================
# Agent
# =========================
def build_agent():
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    tools = [
        analyze_ablation_lesions,
        convert_pixels_to_mm,
        describe_microscopy_image,
        extract_mask_stats,
        search_papers_for_image,
    ]
    return create_agent(model=llm, tools=tools)


def run_agent(agent, user_input: str) -> str:
    result = agent.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
    })

    msgs_out = result.get("messages", [])

    if not msgs_out:
        return "Error: No response generated."

    last = msgs_out[-1]
    content = last.content if hasattr(last, "content") else str(last)

    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    return str(content).strip()


# =========================
# CLI / REPL
# =========================
def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found. Set it in your .env file.")

    print("=" * 60)
    print("Lab Image Analysis Agent")
    print("Commands:")
    print('  analyze "<path>"         - describe any image')
    print('  lesions "<path>"         - detect and measure ablation lesions')
    print('  lesions "<path>" <scale> - detect lesions with manual mm/pixel scale')
    print('  stats "<mask_path>"      - segmentation mask stats')
    print("  q                        - quit")
    print("Or type any natural language query.")
    print("=" * 60)

    agent = build_agent()

    while True:
        try:
            user_input = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        lower = user_input.lower()

        # Command: analyze
        if lower.startswith("analyze "):
            path = clean_wrapping_quotes(user_input[8:].strip())
            try:
                output = describe_microscopy_image.invoke({"image_path": path})
                print("\n--- Agent Response ---")
                print(output)
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Command: lesions
        elif lower.startswith("lesions "):
            raw_args = user_input[8:].strip()
            path, scale = parse_lesions_command_args(raw_args)

            if not path:
                print("Error: provide a file path.")
                continue

            try:
                if scale is not None:
                    output = analyze_ablation_lesions.invoke(
                        {"image_path": path, "mm_per_pixel": scale}
                    )
                else:
                    output = analyze_ablation_lesions.invoke({"image_path": path})

                print("\n--- Agent Response ---")
                print(output)
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Command: stats
        elif lower.startswith("stats "):
            path = clean_wrapping_quotes(user_input[6:].strip())
            try:
                output = extract_mask_stats.invoke({"mask_path": path})
                print("\n--- Mask Stats ---")
                print(output)
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Entire input is just an image path -> direct lesion analysis
        direct_path = extract_file_path_from_text(user_input)
        cleaned_direct = clean_wrapping_quotes(user_input)
        if direct_path and cleaned_direct == direct_path:
            try:
                output = analyze_ablation_lesions.invoke({"image_path": direct_path})
                print("\n--- Agent Response ---")
                print(output)
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Natural-language lesion request with file path -> bypass full agent
        if is_lesion_related_request(user_input):
            detected_path = extract_file_path_from_text(user_input)
            scale_value = extract_scale_from_text(user_input)

            if detected_path:
                try:
                    if scale_value is not None:
                        output = analyze_ablation_lesions.invoke(
                            {"image_path": detected_path, "mm_per_pixel": scale_value}
                        )
                    else:
                        output = analyze_ablation_lesions.invoke(
                            {"image_path": detected_path}
                        )

                    print("\n--- Agent Response ---")
                    print(output)
                except Exception as e:
                    print(f"Error: {e}")
                continue

        # Fallback: full agent, but only return final message
        try:
            output = run_agent(agent, user_input)
            print("\n--- Agent Response ---")
            print(output)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()