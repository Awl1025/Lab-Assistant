"""
AI agent for lab image analysis.
Capabilities:
  1. Describe/analyze many types of images (Gemini vision)
  2. Extract stats from segmentation masks
  3. Search related scientific papers
  4. Analyze cardiac ablation lesions - count, measure width/depth/area
     with ruler and size detection and length scaling from pixels to mm.
"""

import os
import base64
import json
import re
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

load_dotenv()


# Config
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "models/text-embedding-004"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K_PAPERS = 5

SYSTEM_PROMPT = """You are an expert biomedical image analysis assistant with deep knowledge of
cell biology, fluorescence microscopy, cardiac ablation procedures, and bioimage analysis.

You have access to these tools:
- describe_microscopy_image: Visually analyze any lab/microscopy image
- extract_mask_stats: Compute quantitative stats from a segmentation mask
- search_papers_for_image: Find relevant scientific papers
- analyze_ablation_lesions: Detect and measure cardiac ablation lesions, with optional
  ruler-based mm scaling. Use this whenever the user mentions ablation, lesions, or heart tissue.

Workflow guidance:
- For ablation/cardiac images: use analyze_ablation_lesions first.
- For microscopy images: use describe_microscopy_image, then search_papers_for_image.
- For masks: use extract_mask_stats.
- Always report measurements clearly, noting whether units are mm or pixels.
- If a ruler is detected, always use mm. If not, report pixels and offer to convert."""


# Helpers
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
        tmp = "/tmp/_agent_tmp.png"
        img.save(tmp, format="PNG")
        return encode_image_base64(tmp)
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
        pass
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def get_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def ask_gemini_vision(b64: str, prompt: str, json_mode: bool = False) -> str:
    """Send an image + prompt to Gemini and return text response."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }
    resp = llm.invoke([message])
    return resp.content


def draw_lesion_annotations(image_path: Path, lesions: list, scale_info: dict) -> str:
    """
    Draw bounding boxes and labels on the image for each detected lesion.
    Returns path to annotated image.
    """
    img = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = ["#FF4444", "#FF8800", "#FFDD00", "#44FF44", "#4488FF",
              "#AA44FF", "#FF44AA", "#00FFFF", "#FF6666", "#88FF88"]

    for i, lesion in enumerate(lesions):
        color = colors[i % len(colors)]
        # bbox expected as [x1_frac, y1_frac, x2_frac, y2_frac] (0..1 relative)
        bbox = lesion.get("bbox_relative")
        if not bbox:
            continue
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        units = scale_info.get("unit", "px")
        width_val  = lesion.get(f"width_{units}", lesion.get("width_px", "?"))
        depth_val  = lesion.get(f"depth_{units}", lesion.get("depth_px", "?"))

        label = f"L{i+1} W:{width_val} D:{depth_val} {units}"
        draw.rectangle([x1, max(0, y1-18), x1+len(label)*7+4, y1], fill=color)
        draw.text((x1+2, max(0, y1-16)), label, fill="black")

    out_path = f"/tmp/_annotated_{image_path.stem}.png"
    img.save(out_path)
    return out_path



# Agent Tools
@tool
def analyze_ablation_lesions(image_path: str, mm_per_pixel: float = 0.0) -> str:
    """
    Detect and measure cardiac ablation lesions in a gross pathology image.
    Identifies lesion count, width, depth, and area per lesion.
    Automatically detects a ruler for mm scaling; falls back to pixels.
    Use mm_per_pixel to manually override scale (e.g. 0.1 means 1px = 0.1mm).
    image_path: path to the image file.
    mm_per_pixel: optional manual scale override (0 = auto-detect).
    """
    path = Path(image_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    b64 = prep_image_for_gemini(path)
    img = Image.open(str(path))
    img_w, img_h = img.size

    # --- Step 1: Detect ruler and estimate scale ---
    ruler_prompt = """Examine this cardiac ablation gross pathology image carefully.

Is there a ruler, scale bar, or measuring device visible in the image?
If YES: 
  - Estimate what portion of the image width the ruler spans (as a fraction 0.0-1.0)
  - Estimate the total length the ruler represents in mm
  - Estimate where the ruler is located (top/bottom/left/right edge)

Respond ONLY in this exact JSON format (no markdown, no extra text):
{
  "ruler_found": true or false,
  "ruler_span_fraction": 0.0,
  "ruler_length_mm": 0,
  "ruler_location": "bottom-right",
  "confidence": "high/medium/low"
}"""

    ruler_raw = ask_gemini_vision(b64, ruler_prompt)
    ruler_raw = re.sub(r"```json|```", "", ruler_raw).strip()

    scale_info = {"unit": "px", "mm_per_pixel": None}
    ruler_note = "No ruler detected — measurements in pixels."

    try:
        ruler_data = json.loads(ruler_raw)
        if ruler_data.get("ruler_found") and ruler_data.get("ruler_length_mm", 0) > 0:
            span_frac = float(ruler_data.get("ruler_span_fraction", 0.5))
            ruler_px  = span_frac * img_w
            mpp       = float(ruler_data["ruler_length_mm"]) / ruler_px if ruler_px > 0 else 0
            scale_info = {"unit": "mm", "mm_per_pixel": mpp}
            ruler_note = (
                f"Ruler detected ({ruler_data.get('confidence','?')} confidence): "
                f"{ruler_data['ruler_length_mm']}mm spans ~{ruler_px:.0f}px → "
                f"{mpp:.4f} mm/pixel"
            )
    except Exception:
        pass

    # Manual override
    if mm_per_pixel > 0:
        scale_info = {"unit": "mm", "mm_per_pixel": mm_per_pixel}
        ruler_note = f"Manual scale applied: {mm_per_pixel} mm/pixel"

    mpp = scale_info.get("mm_per_pixel") or 1.0
    unit = scale_info["unit"]

    # --- Step 2: Detect and measure lesions ---
    lesion_prompt = f"""This is a cardiac ablation gross pathology image (cross-section of heart tissue).
The image is {img_w} x {img_h} pixels.

Identify ALL ablation lesions visible. Ablation lesions appear as:
- Dark brown or tan-brown discolored regions in the myocardium
- Often circular or oval in cross-section
- May show a pale white center (coagulation necrosis) with darker brown border
- Located within the heart wall tissue

For EACH lesion provide a bounding box as fractions of image dimensions (0.0 to 1.0).
Also estimate the lesion's width and depth in pixels by examining its extent.

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "lesion_count": 3,
  "lesions": [
    {{
      "id": 1,
      "description": "dark brown oval lesion, upper left",
      "bbox_relative": [x1, y1, x2, y2],
      "width_px": 120,
      "depth_px": 85,
      "shape": "oval/circular/irregular",
      "confidence": "high/medium/low"
    }}
  ],
  "tissue_notes": "general observations about the tissue"
}}"""

    lesion_raw = ask_gemini_vision(b64, lesion_prompt)
    lesion_raw = re.sub(r"```json|```", "", lesion_raw).strip()

    try:
        lesion_data = json.loads(lesion_raw)
    except Exception as e:
        return (
            f"Scale: {ruler_note}\n\n"
            f"Lesion detection failed to parse structured response.\n"
            f"Raw Gemini output:\n{lesion_raw}"
        )

    lesions = lesion_data.get("lesions", [])
    count   = lesion_data.get("lesion_count", len(lesions))

    # --- Step 3: Build report ---
    lines = [
        "=" * 55,
        "  CARDIAC ABLATION LESION ANALYSIS REPORT",
        "=" * 55,
        f"  Image       : {path.name}  ({img_w} x {img_h} px)",
        f"  Scale       : {ruler_note}",
        f"  Lesions found: {count}",
        "-" * 55,
    ]

    total_area = 0.0
    for lesion in lesions:
        lid       = lesion.get("id", "?")
        w_px      = float(lesion.get("width_px", 0))
        d_px      = float(lesion.get("depth_px", 0))
        area_px   = np.pi * (w_px / 2) * (d_px / 2)  # ellipse approximation
        desc      = lesion.get("description", "")
        shape     = lesion.get("shape", "unknown")
        conf      = lesion.get("confidence", "?")

        if unit == "mm":
            w_mm     = w_px * mpp
            d_mm     = d_px * mpp
            area_mm2 = area_px * (mpp ** 2)
            total_area += area_mm2
            lines += [
                f"  Lesion {lid}: {desc}",
                f"    Shape      : {shape}  (confidence: {conf})",
                f"    Width      : {w_mm:.2f} mm  ({w_px:.0f} px)",
                f"    Depth      : {d_mm:.2f} mm  ({d_px:.0f} px)",
                f"    Area       : {area_mm2:.2f} mm²  (ellipse approx)",
                "",
            ]
        else:
            total_area += area_px
            lines += [
                f"  Lesion {lid}: {desc}",
                f"    Shape      : {shape}  (confidence: {conf})",
                f"    Width      : {w_px:.0f} px",
                f"    Depth      : {d_px:.0f} px",
                f"    Area       : {area_px:.0f} px²  (ellipse approx)",
                f"    (Tip: provide mm_per_pixel to convert to mm)",
                "",
            ]

    lines += [
        "-" * 55,
        f"  Total lesion area : {total_area:.2f} {unit}²" if unit == "mm" else f"  Total lesion area : {total_area:.0f} px²",
        f"  Tissue notes      : {lesion_data.get('tissue_notes', 'N/A')}",
        "=" * 55,
    ]

    # --- Step 4: Save annotated image ---
    try:
        ann_path = draw_lesion_annotations(path, lesions, scale_info)
        lines.append(f"\n  Annotated image saved to: {ann_path}")
    except Exception as e:
        lines.append(f"\n  (Annotation failed: {e})")

    return "\n".join(lines)


@tool
def convert_pixels_to_mm(pixels: float, mm_per_pixel: float) -> str:
    """
    Convert a pixel measurement to millimeters using a known scale ratio.
    pixels: measurement in pixels.
    mm_per_pixel: scale factor (e.g. 0.1 means 1 pixel = 0.1 mm).
    """
    mm = pixels * mm_per_pixel
    area_mm2 = np.pi * (pixels / 2) ** 2 * (mm_per_pixel ** 2)
    return (
        f"Conversion using scale: {mm_per_pixel} mm/pixel\n"
        f"  {pixels:.1f} px  →  {mm:.3f} mm\n"
        f"  If circular: area = {area_mm2:.3f} mm²"
    )


@tool
def describe_microscopy_image(image_path: str) -> str:
    """
    Analyze any lab/microscopy image using Gemini vision.
    Describe morphology, density, staining patterns, and notable features.
    image_path: path to a .png, .jpg, .tif, or .tiff file.
    """
    path = Path(image_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    b64 = prep_image_for_gemini(path)
    prompt = (
        "You are an expert bioimage analyst. Analyze this lab image and provide:\n"
        "1. Tissue/cell morphology (shape, size, texture, color)\n"
        "2. Approximate density or distribution of structures\n"
        "3. Staining or visual pattern observations\n"
        "4. Any abnormalities or notable features\n"
        "5. Suggested analysis approach\n"
        "Be concise and scientifically precise."
    )
    return ask_gemini_vision(b64, prompt)


@tool
def extract_mask_stats(mask_path: str) -> str:
    """
    Extract quantitative statistics from a binary segmentation mask.
    Returns object count, foreground coverage %, and size stats.
    mask_path: path to mask image (.png, .tif, .tiff).
    """
    from scipy import ndimage as ndi

    path = Path(mask_path.strip())
    if not path.exists():
        return f"Error: File not found at '{path}'."

    mask = load_mask_array(str(path))
    total_pixels     = mask.size
    foreground_pixels = int(mask.sum())
    coverage_pct     = round(100.0 * foreground_pixels / total_pixels, 2)

    labeled, num_objects = ndi.label(mask)
    sizes = np.array(ndi.sum(mask, labeled, range(1, num_objects + 1)))

    mean_size   = round(float(sizes.mean()), 1)    if len(sizes) > 0 else 0.0
    median_size = round(float(np.median(sizes)), 1) if len(sizes) > 0 else 0.0
    min_size    = int(sizes.min()) if len(sizes) > 0 else 0
    max_size    = int(sizes.max()) if len(sizes) > 0 else 0

    return (
        f"Mask Statistics: {path.name}\n"
        f"  Image size       : {mask.shape[1]} x {mask.shape[0]} px\n"
        f"  Detected objects : {num_objects}\n"
        f"  Foreground       : {foreground_pixels} px  ({coverage_pct}%)\n"
        f"  Object sizes     : mean={mean_size} | median={median_size} | "
        f"min={min_size} | max={max_size} px"
    )


@tool
def search_papers_for_image(query: str) -> str:
    """
    Search the scientific paper vector store for content related to a query.
    query: keywords about what was observed in the image.
    """
    db = get_vectorstore()
    if db is None:
        return (
            "No vector store found at './chroma_db'. "
            "Run lab-assistant.py first to build the paper index."
        )

    results = db.similarity_search(query, k=TOP_K_PAPERS)
    if not results:
        return "No relevant papers found for that query."

    lines = [f"Top {len(results)} paper excerpts for: '{query}'\n"]
    for i, doc in enumerate(results, start=1):
        title    = doc.metadata.get("title", "Unknown")
        dataset  = doc.metadata.get("dataset", "Unknown")
        paper_id = doc.metadata.get("paper_id") or doc.metadata.get("pmid", "?")
        snippet  = doc.page_content[:400].replace("\n", " ")
        lines.append(f"{i}. [{dataset}] {title} (ID: {paper_id})\n   ...{snippet}...\n")
    return "\n".join(lines)



# Agent
def build_agent():
    llm   = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    tools = [
        analyze_ablation_lesions,
        convert_pixels_to_mm,
        describe_microscopy_image,
        extract_mask_stats,
        search_papers_for_image,
    ]
    return create_agent(model=llm, tools=tools)


def run_agent(agent, user_input: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]
    result = agent.invoke({"messages": messages})
    msgs_out = result.get("messages", [])
    if msgs_out:
        last    = msgs_out[-1]
        content = last.content if hasattr(last, "content") else str(last)
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content)
        return content
    return str(result)



# REPL
def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found. Set it in your .env file.")

    print("=" * 60)
    print("  Lab Image Analysis Agent")
    print("  Commands:")
    print("    analyze <path>         - describe any image")
    print("    lesions <path>         - detect & measure ablation lesions")
    print("    lesions <path> <scale> - with manual mm/pixel scale")
    print("    stats <mask_path>      - segmentation mask stats")
    print("    q                      - quit")
    print("  Or type any natural language query.")
    print("=" * 60)

    agent = build_agent()

    while True:
        try:
            user_input = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        # Convenience shortcuts
        lower = user_input.lower()
        if lower.startswith("analyze "):
            path = user_input[8:].strip()
            user_input = f"Please analyze this image: {path}"
        elif lower.startswith("lesions "):
            parts = user_input[8:].strip().split()
            path  = parts[0]
            scale = parts[1] if len(parts) > 1 else "0"
            user_input = (
                f"Analyze ablation lesions in this image: {path}"
                + (f" Use mm_per_pixel={scale}" if scale != "0" else "")
            )
        elif lower.startswith("stats "):
            path = user_input[6:].strip()
            user_input = f"Extract segmentation statistics from this mask: {path}"

        try:
            output = run_agent(agent, user_input)
            print("\n--- Agent Response ---")
            print(output)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
