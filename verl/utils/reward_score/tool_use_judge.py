# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import re
import ast
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI, DefaultHttpxClient
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore
    DefaultHttpxClient = None  # type: ignore


logger = logging.getLogger(__name__)


TRANSFORM_KEYS = {
    "crop",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
}


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _safe_eval_numeric(expr: str) -> Optional[float]:
    """Safely evaluate a simple numeric expression.

    Supports literals and + - * / // % with parentheses and unary +/-.
    Returns None if not safely evaluable.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(n) -> Optional[float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return float(n.n)  # type: ignore[attr-defined]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            if val is None:
                return None
            return +val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
        ):
            left = _eval(n.left)
            right = _eval(n.right)
            if left is None or right is None:
                return None
            try:
                if isinstance(n.op, ast.Add):
                    return left + right
                if isinstance(n.op, ast.Sub):
                    return left - right
                if isinstance(n.op, ast.Mult):
                    return left * right
                if isinstance(n.op, ast.Div):
                    return left / right
                if isinstance(n.op, ast.FloorDiv):
                    return float(left // right)
                return left % right
            except Exception:
                return None
        return None

    val = _eval(node)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _safe_eval_node(n: ast.AST, env: Dict[str, Any]) -> Optional[Any]:
    """Safely evaluate an AST node using a simple numeric environment.

    Supports numeric literals, names, tuples/lists, basic arithmetic, and
    safe builtin calls: min, max, int, float, round, abs, tuple, list.
    Returns None if not safely evaluable.
    """
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return float(n.value)
    if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
        return float(getattr(n, "n"))  # type: ignore[attr-defined]
    if isinstance(n, ast.Name):
        return env.get(n.id)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        val = _safe_eval_node(n.operand, env)
        if val is None:
            return None
        try:
            v = float(val)
            return +v if isinstance(n.op, ast.UAdd) else -v
        except Exception:
            return None
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
        left = _safe_eval_node(n.left, env)
        right = _safe_eval_node(n.right, env)
        if left is None or right is None:
            return None
        try:
            lf = float(left)
            rf = float(right)
            if isinstance(n.op, ast.Add):
                return lf + rf
            if isinstance(n.op, ast.Sub):
                return lf - rf
            if isinstance(n.op, ast.Mult):
                return lf * rf
            if isinstance(n.op, ast.Div):
                return lf / rf
            if isinstance(n.op, ast.FloorDiv):
                return float(lf // rf)
            return lf % rf
        except Exception:
            return None
    if isinstance(n, (ast.Tuple, ast.List)):
        vals: List[Optional[Any]] = []
        for elt in n.elts:  # type: ignore[attr-defined]
            vals.append(_safe_eval_node(elt, env))
        if any(v is None for v in vals):
            return None
        try:
            return [float(v) for v in vals]  # type: ignore[list-item]
        except Exception:
            return None
    if isinstance(n, ast.Call):
        func_name = None
        if isinstance(n.func, ast.Name):
            func_name = n.func.id
        args: List[Any] = []
        for a in n.args:
            args.append(_safe_eval_node(a, env))
        if func_name in ("int", "float", "abs"):
            if len(args) >= 1 and args[0] is not None:
                try:
                    x = float(args[0])
                    if func_name == "int":
                        return float(int(x))
                    if func_name == "float":
                        return float(x)
                    return abs(x)
                except Exception:
                    return None
            return None
        if func_name == "round":
            if len(args) >= 1 and args[0] is not None:
                try:
                    x = float(args[0])
                    if len(args) >= 2 and args[1] is not None:
                        nd = int(float(args[1]))
                        return float(round(x, nd))
                    return float(round(x))
                except Exception:
                    return None
            return None
        if func_name in ("min", "max"):
            if not args or any(a is None for a in args):
                return None
            try:
                vals_num = [float(a) for a in args]
                return float(min(vals_num) if func_name == "min" else max(vals_num))
            except Exception:
                return None
        if func_name in ("tuple", "list"):
            if len(args) == 1 and isinstance(n.args[0], (ast.List, ast.Tuple)):
                inner = _safe_eval_node(n.args[0], env)
                if isinstance(inner, list) and all(isinstance(x, (int, float)) for x in inner):
                    return [float(x) for x in inner]
            if args and all(a is not None for a in args):
                try:
                    return [float(a) for a in args]  # type: ignore[list-item]
                except Exception:
                    return None
            return None
        return None
    return None


def _build_env_before(code: str, line_no: int) -> Dict[str, Any]:
    """Build a simple numeric environment by replaying assignments before a given line."""
    env: Dict[str, Any] = {}
    try:
        module = ast.parse(code)
    except Exception:
        return env
    items: List[Tuple[int, int, ast.AST]] = []
    for node in ast.walk(module):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and hasattr(node, "lineno"):
            items.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), node))
    items.sort(key=lambda x: (x[0], x[1]))
    for ln, _, node in items:
        if ln >= line_no:
            break
        try:
            if isinstance(node, ast.Assign):
                value = _safe_eval_node(node.value, env)
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        if value is not None:
                            env[tgt.id] = value
                    elif isinstance(tgt, (ast.Tuple, ast.List)):
                        if value is not None and isinstance(value, list):
                            names: List[str] = []
                            for elt in tgt.elts:  # type: ignore[attr-defined]
                                if isinstance(elt, ast.Name):
                                    names.append(elt.id)
                            if len(names) == len(value):
                                for n_name, n_val in zip(names, value, strict=False):
                                    try:
                                        env[n_name] = float(n_val)
                                    except Exception:
                                        pass
            elif isinstance(node, ast.AnnAssign):
                if node.value is None or not isinstance(node.target, ast.Name):
                    continue
                val = _safe_eval_node(node.value, env)
                if val is not None:
                    env[node.target.id] = val
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                cur = env.get(node.target.id)
                right = _safe_eval_node(node.value, env)
                if cur is None or right is None:
                    continue
                try:
                    cf = float(cur)
                    rf = float(right)
                    if isinstance(node.op, ast.Add):
                        env[node.target.id] = cf + rf
                    elif isinstance(node.op, ast.Sub):
                        env[node.target.id] = cf - rf
                    elif isinstance(node.op, ast.Mult):
                        env[node.target.id] = cf * rf
                    elif isinstance(node.op, ast.Div):
                        env[node.target.id] = cf / rf
                    elif isinstance(node.op, ast.FloorDiv):
                        env[node.target.id] = float(cf // rf)
                    elif isinstance(node.op, ast.Mod):
                        env[node.target.id] = cf % rf
                except Exception:
                    pass
        except Exception:
            continue
    return env


def _extract_crop_boxes_from_ast(code: str) -> List[List[float]]:
    """Extract crop boxes by parsing AST and evaluating arguments and prior assignments."""
    boxes: List[List[float]] = []
    try:
        module = ast.parse(code)
        call_nodes: List[Tuple[int, int, ast.Call]] = []
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and hasattr(node, "lineno"):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "crop":
                    call_nodes.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), node))
        call_nodes.sort(key=lambda x: (x[0], x[1]))
        for ln, _, call in call_nodes:
            if not call.args:
                continue
            env = _build_env_before(code, ln)
            arg0 = call.args[0]
            vals = _safe_eval_node(arg0, env)
            if isinstance(vals, list) and len(vals) == 4 and all(isinstance(v, (int, float)) for v in vals):
                try:
                    box = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]
                    boxes.append(box)
                except Exception:
                    pass
    except Exception:
        pass
    return boxes


def _regex_literal_crops(code: str) -> List[List[float]]:
    """Regex fallback for literal crop((x1,y1,x2,y2)) boxes."""
    boxes: List[List[float]] = []
    for m in re.finditer(r"\.crop\(\(\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*\)\)", code):
        try:
            x1, y1, x2, y2 = map(float, m.groups())
            boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
    return boxes


def _dedup_boxes(boxes: List[List[float]]) -> List[List[float]]:
    seen = set()
    uniq: List[List[float]] = []
    for b in boxes:
        try:
            key = (round(float(b[0]), 6), round(float(b[1]), 6), round(float(b[2]), 6), round(float(b[3]), 6))
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            uniq.append([float(key[0]), float(key[1]), float(key[2]), float(key[3])])
    return uniq


def _rotations_from_code(code: str) -> List[str]:
    used: List[str] = []
    for m in re.finditer(r"\.rotate\(\s*([^,\)]+)\s*(?:,\s*[^)]*)?\)", code):
        try:
            expr = m.group(1).strip()
            angle_val = _safe_eval_numeric(expr)
            if angle_val is None:
                continue
            angle = float(angle_val) % 360
            def near(a: float, b: float, tol: float = 5.0) -> bool:
                return abs((a - b + 180) % 360 - 180) <= tol
            if near(angle, 90):
                used.append("rotate_90")
            elif near(angle, 180):
                used.append("rotate_180")
            elif near(angle, 270):
                used.append("rotate_270")
        except Exception:
            pass
    if re.search(r"transpose\(\s*Image\.ROTATE_90\s*\)", code):
        used.append("rotate_90")
    if re.search(r"transpose\(\s*Image\.ROTATE_180\s*\)", code):
        used.append("rotate_180")
    if re.search(r"transpose\(\s*Image\.ROTATE_270\s*\)", code):
        used.append("rotate_270")
    for m in re.finditer(r"np\.rot90\(\s*[^,\s)]+\s*(?:,\s*k\s*=\s*([-+]?\d+))?", code):
        try:
            k_str = m.group(1)
            k = int(k_str) if k_str is not None else 1
            k = k % 4
            if k == 1:
                used.append("rotate_90")
            elif k == 2:
                used.append("rotate_180")
            elif k == 3:
                used.append("rotate_270")
        except Exception:
            pass
    if re.search(r"cv2\.rotate\(\s*[^,)]*\s*,\s*cv2\.ROTATE_90_CLOCKWISE\s*\)", code):
        used.append("rotate_270")
    if re.search(r"cv2\.rotate\(\s*[^,)]*\s*,\s*cv2\.ROTATE_180\s*\)", code):
        used.append("rotate_180")
    if re.search(r"cv2\.rotate\(\s*[^,)]*\s*,\s*cv2\.ROTATE_90_COUNTERCLOCKWISE\s*\)", code):
        used.append("rotate_90")
    return used


def _flips_from_code(code: str) -> List[str]:
    used: List[str] = []
    if re.search(r"ImageOps\.mirror|transpose\(\s*Image\.FLIP_LEFT_RIGHT\s*\)", code):
        used.append("flip_horizontal")
    if re.search(r"ImageOps\.flip|transpose\(\s*Image\.FLIP_TOP_BOTTOM\s*\)", code):
        used.append("flip_vertical")
    return used


def _other_transforms_from_code(code: str) -> List[str]:
    """Heuristic detectors for non-rotate/flip/crop transforms.

    Returns a list that may include: resize, grayscale, autocontrast, equalize, invert,
    blur, gaussian_blur, median_blur, sharpen, edge_enhance, edge_detect,
    brightness_up/down, contrast_up/down, sharpness_up/down, color_up/down.
    """
    used: List[str] = []

    # 1) PIL ImageEnhance.*(...).enhance(factor)
    for m in re.finditer(r"ImageEnhance\.(Brightness|Contrast|Sharpness|Color)\s*\(\s*[^)]+\s*\)\s*\.enhance\(\s*([^\)]+?)\s*\)", code):
        kind = m.group(1)
        expr = m.group(2).strip()
        factor = _safe_eval_numeric(expr)
        base = {
            "Brightness": "brightness",
            "Contrast": "contrast",
            "Sharpness": "sharpness",
            "Color": "color",
        }[kind]
        if factor is None:
            used.append(base)
        else:
            try:
                f = float(factor)
                # Small tolerance band to avoid noise
                if f > 1.05:
                    used.append(f"{base}_up")
                elif f < 0.95:
                    used.append(f"{base}_down")
                else:
                    used.append(base)
            except Exception:
                used.append(base)

    # 2) PIL Image.filter(ImageFilter.*)
    # GaussianBlur(radius), BoxBlur(radius), MedianFilter(size), UnsharpMask, BLUR/SHARPEN/SMOOTH/DETAIL, EDGE_*.
    for m in re.finditer(r"ImageFilter\.([A-Za-z_]+)(?:\s*\(\s*([^\)]*)\s*\))?", code):
        name = m.group(1)
        name_upper = name.upper()
        if name_upper in {"BLUR", "BOXBLUR"}:
            used.append("blur")
        elif name_upper in {"GAUSSIANBLUR"}:
            used.append("gaussian_blur")
        elif name_upper in {"MEDIANFILTER"}:
            used.append("median_blur")
        elif name_upper in {"SHARPEN", "UNSHARPMASK"}:
            used.append("sharpen")
        elif name_upper in {"SMOOTH", "SMOOTH_MORE"}:
            used.append("smooth")
        elif name_upper in {"DETAIL"}:
            used.append("detail")
        elif name_upper.startswith("EDGE_ENHANCE"):
            used.append("edge_enhance")
        elif name_upper in {"FIND_EDGES"}:
            used.append("edge_detect")

    # 3) Grayscale conversions
    if re.search(r"\.convert\(\s*['\"]L['\"]\s*\)", code):
        used.append("grayscale")
    if re.search(r"cv2\.cvtColor\(\s*[^,]+,\s*cv2\.COLOR_(?:BGR|RGB)2GRAY\s*\)", code):
        used.append("grayscale")

    # 4) Resize
    if re.search(r"\.resize\(\s*\(\s*[^\)]+\)\s*\)", code):
        used.append("resize")
    if re.search(r"ImageOps\.fit\(\s*[^,]+,\s*\(\s*[^\)]+\)\s*\)", code):
        used.append("resize")

    # 5) ImageOps helpers
    if re.search(r"ImageOps\.autocontrast\(\s*[^\)]*\)", code):
        used.append("autocontrast")
    if re.search(r"ImageOps\.equalize\(\s*[^\)]*\)", code):
        used.append("equalize")
    if re.search(r"ImageOps\.invert\(\s*[^\)]*\)", code):
        used.append("invert")
    if re.search(r"ImageOps\.solarize\(\s*[^\)]*\)", code):
        used.append("solarize")
    if re.search(r"ImageOps\.posterize\(\s*[^\)]*\)", code):
        used.append("posterize")

    # 6) OpenCV blurs and edges
    if re.search(r"cv2\.GaussianBlur\(\s*[^\)]*\)", code):
        used.append("gaussian_blur")
    if re.search(r"cv2\.(?:blur|boxFilter)\(\s*[^\)]*\)", code):
        used.append("blur")
    if re.search(r"cv2\.medianBlur\(\s*[^\)]*\)", code):
        used.append("median_blur")
    if re.search(r"cv2\.Canny\(\s*[^\)]*\)", code):
        used.append("edge_detect")
    if re.search(r"cv2\.Sobel\(\s*[^\)]*\)\s*|\s*cv2\.Laplacian\(\s*[^\)]*\)", code):
        used.append("edge_detect")

    # 7) OpenCV brightness/contrast via convertScaleAbs(..., alpha=?, beta=?)
    for m in re.finditer(r"cv2\.convertScaleAbs\(\s*[^\)]*\)", code):
        seg = m.group(0)
        m_alpha = re.search(r"alpha\s*=\s*([-+]?\d+(?:\.\d+)?)", seg)
        m_beta = re.search(r"beta\s*=\s*([-+]?\d+(?:\.\d+)?)", seg)
        if m_alpha:
            try:
                a = float(m_alpha.group(1))
                if a > 1.05:
                    used.append("contrast_up")
                elif a < 0.95:
                    used.append("contrast_down")
                else:
                    used.append("contrast")
            except Exception:
                used.append("contrast")
        if m_beta:
            try:
                b = float(m_beta.group(1))
                if b > 0.5:
                    used.append("brightness_up")
                elif b < -0.5:
                    used.append("brightness_down")
                else:
                    used.append("brightness")
            except Exception:
                used.append("brightness")

    # 8) Morphology
    if re.search(r"cv2\.erode\(\s*[^\)]*\)", code):
        used.append("erode")
    if re.search(r"cv2\.dilate\(\s*[^\)]*\)", code):
        used.append("dilate")
    if re.search(r"cv2\.equalizeHist\(\s*[^\)]*\)", code):
        used.append("equalize")

    return _unique_preserve_order(used)


def _normalize_description(description: Any) -> str:
    if description is None:
        return ""
    if isinstance(description, dict):
        for key in ["description", "text", "content", "summary"]:
            if key in description and isinstance(description[key], str):
                return description[key]
        return ""
    if isinstance(description, str):
        return description
    return str(description)


def _transforms_from_description(description: Any) -> List[str]:
    desc = _normalize_description(description).lower()
    used: List[str] = []
    if "horizontal" in desc and ("flip" in desc or "mirror" in desc):
        used.append("flip_horizontal")
    if "vertical" in desc and "flip" in desc:
        used.append("flip_vertical")
    if "crop" in desc:
        used.append("crop")
    return used


def _boxes_from_description(description: Any) -> List[List[float]]:
    desc = _normalize_description(description)
    boxes: List[List[float]] = []
    m = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", desc)
    if m:
        try:
            x1, y1, x2, y2 = map(float, m.groups())
            boxes.append([x1, y1, x2, y2])
            return boxes
        except Exception:
            boxes = []
    m = re.search(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", desc)
    if m:
        try:
            x1, y1, x2, y2 = map(float, m.groups())
            boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
    return boxes


class ToolUseJudgeClient:
    """Classifies which image-transforms are used in code_image_tool invocations.

    It extracts a normalized set of transforms and any crop bounding boxes found.
    Transforms include: crop, rotate_90/180/270, flip_horizontal/vertical.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (
            os.getenv("TOOL_JUDGE_BASE_URL")
            or base_url
            or os.getenv("LLM_JUDGE_BASE_URL", "")
        )
        self.model_name = (
            os.getenv("TOOL_JUDGE_MODEL_NAME")
            or model_name
            or os.getenv("LLM_JUDGE_MODEL_NAME", None)
        )
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = None
        if self.base_url and OpenAI is not None:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    http_client=DefaultHttpxClient(trust_env=False, timeout=timeout),
                    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                )
                if self.model_name is None:
                    # Attempt to auto-detect a model
                    self.model_name = self._get_model_name()
                print(f"[INFO]: ToolUseJudgeClient initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning("Failed to init ToolUseJudgeClient: %s", e)
                self.client = None

    def _get_model_name(self) -> Optional[str]:
        try:
            import requests

            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(f"{self.base_url}/models", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("data"):
                return data["data"][0]["id"]
        except Exception as e:
            logger.debug("ToolUseJudgeClient model discovery failed: %s", e)
        return None

    @staticmethod
    def heuristic_parse(code: str, description: str = "") -> Tuple[List[str], List[List[float]]]:
        """Lightweight parsing: delegate to helpers for transforms and boxes."""
        if code is None:
            code = ""
        crop_boxes = _dedup_boxes(_extract_crop_boxes_from_ast(code) + _regex_literal_crops(code))
        used = _rotations_from_code(code) + _flips_from_code(code) + _other_transforms_from_code(code)
        if crop_boxes:
            used.append("crop")
        used = _unique_preserve_order(used + _transforms_from_description(description))
        if not crop_boxes:
            crop_boxes = _boxes_from_description(description)
        return used, crop_boxes

    def analyze(self, code: str, description: str = "") -> Dict[str, Any]:
        """Analyze with heuristic first, then optionally use LLM to refine.

        Returns dict with keys: transforms: List[str], crop_boxes: List[List[float]]
        """
        # Handle None code parameter
        if code is None:
            code = ""
        elif not isinstance(code, str):
            code = str(code)
            
        # Handle different types of description parameter
        if description is None:
            description = ""
        elif isinstance(description, dict):
            # If description is a dict, try to extract text from common keys
            description = ""
            for key in ["description", "text", "content", "summary"]:
                if key in description and isinstance(description[key], str):
                    description = description[key]
                    break
        elif not isinstance(description, str):
            description = str(description)
        transforms, crop_boxes = self.heuristic_parse(code, description)

        if self.client is None:
            # print("[DEBUG]: No LLM client found, using heuristic only")
            return {"transforms": transforms, "crop_boxes": crop_boxes}

        # If we already detected things, we can skip LLM to save cost
        if transforms:
            return {"transforms": transforms, "crop_boxes": crop_boxes}
        
        # print("[DEBUG]: Using LLM to analyze tool use")

        system_prompt = (
            "You are a precise code analyzer. Given Python code that operates on a PIL image, "
            "identify which of the following transforms are used: crop, rotate_90, rotate_180, rotate_270, "
            "flip_horizontal, flip_vertical, others.\n"
            "Rotate_angle here is **counter-clockwise** by 'angle' degrees. Treat only rotations "
            "that are approximately 90/180/270 degrees as rotate_90/rotate_180/rotate_270 respectively.\n"
            "If the code crops (zoom in) a rectangular region, extract the bounding box as [x_min,y_min,x_max,y_max].\n"
            "If the code uses other transforms, you should add a specific, short and unique transform name (for example, contrast_up, brightness_down, resize, etc.) to the transforms list.\n"
            "Respond ONLY with compact JSON: {\"transforms\":[...],\"crop_boxes\":[[x_min,y_min,x_max,y_max], ...]} without extra text."
        )
        user_prompt = (
            "Code:\n" + code + "\n\n" +
            ("Description:\n" + description if description else "")
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            # print("[DEBUG]: LLM response: ", content)
            data = json.loads(content)
            t_list = [t for t in data.get("transforms", [])]
            c_list = data.get("crop_boxes", []) or []
            # Ensure numeric boxes
            norm_boxes: List[List[float]] = []
            for box in c_list:
                try:
                    norm_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                except Exception:
                    continue
            # Sanitize LLM result: only accept rotate_90/180/270 if corroborated by heuristics from code
            # Re-run heuristics on code-only (ignore description hints) to avoid false positives
            h_transforms, _ = self.heuristic_parse(code, "")
            h_rotates = {t for t in h_transforms if t in {"rotate_90", "rotate_180", "rotate_270"}}
            filtered_t_list: List[str] = []
            for t in t_list:
                if t in {"rotate_90", "rotate_180", "rotate_270"}:
                    if t in h_rotates:
                        filtered_t_list.append(t)
                    else:
                        # drop spurious rotate classifications
                        continue
                else:
                    filtered_t_list.append(t)
            return {"transforms": _unique_preserve_order(filtered_t_list), "crop_boxes": norm_boxes + crop_boxes}
        except Exception as e:
            logger.debug("ToolUseJudgeClient LLM analyze failed, fallback to heuristic: %s", e)
            return {"transforms": transforms, "crop_boxes": crop_boxes}



