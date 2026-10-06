"""
Computer Vision Agent
----------------------
A Streamlit agent with THREE tools, chosen by a router:

  1. generate_image  -> turns a text prompt into a brand-new image
  2. describe_image   -> analyzes/describes an image the user uploads
  3. edit_image       -> recreates the uploaded image with the user's requested changes

Setup:
1. pip install streamlit requests pillow python-dotenv
2. Get a free Hugging Face API token: https://huggingface.co/settings/tokens
3. Set it as an environment variable before running:
       Windows (PowerShell):  $env:HF_API_TOKEN="hf_xxxxxxxx"
       Windows (cmd):         set HF_API_TOKEN=hf_xxxxxxxxx
   or
   { create a .env file next to this app with:
       HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
       1 : conda activate myenv
       2 : pip install streamlit requests pillow python-dotenv (for only first time)
4. Run:  streamlit run image.py }
"""

import os
import io
import base64
import time
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the same folder as this Python file.
# This is more reliable than load_dotenv() when Streamlit is launched
# from another working directory.
APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------
from huggingface_hub import InferenceClient

def get_hf_token() -> str | None:
    """Get the Hugging Face token from Streamlit secrets, .env, or environment.

    Priority:
      1. Streamlit secrets: HF_API_TOKEN / HF_TOKEN
      2. .env or Windows environment: HF_API_TOKEN / HF_TOKEN
    """
    # Streamlit Cloud / local Streamlit secrets
    try:
        token = st.secrets.get("HF_API_TOKEN") or st.secrets.get("HF_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        # No secrets.toml or secrets is not configured.
        pass

    # Local .env / operating-system environment
    token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    return None


def get_client() -> InferenceClient:
    token = get_hf_token()
    if not token:
        raise AgentError(
            "Missing API token",
            "Hugging Face token not found. Put HF_API_TOKEN=hf_... in "
            f"{ENV_FILE}, or add HF_API_TOKEN to Streamlit secrets, then restart Streamlit.",
        )

    # api_key is the current Hugging Face InferenceClient authentication parameter.
    return InferenceClient(
        api_key=token,
        provider="auto",
        timeout=120,
    )

IMAGE_GEN_MODELS = {
    "FLUX.1-schnell (fast, good quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL (slower, high quality)": "stabilityai/stable-diffusion-xl-base-1.0",
}

CAPTION_MODEL = "Salesforce/blip-image-captioning-large"   # tool 2: describe_image
EDIT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"                   # tool 3: edit_image
ROUTER_MODEL = "facebook/bart-large-mnli"                    # zero-shot classifier for routing

API_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model_id}"
MAX_UPLOAD_MB = 5

st.set_page_config(page_title="Computer Vision Agent", page_icon="🎨", layout="centered")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.main .block-container {padding-top: 2rem; max-width: 780px;}
h1 {font-weight: 800; letter-spacing: -0.5px;}
.agent-subtitle {color: #8a8fa3; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.6rem;}
.tool-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.3px;
    margin-bottom: 10px;
}
.reason-text {color: #8a8fa3; font-size: 0.9rem; margin-bottom: 1rem;}
div[data-testid="stButton"] button {
    border-radius: 10px; font-weight: 600; padding: 0.6rem 0; font-size: 1.02rem;
}
.error-card {
    background: #2b1418; border: 1px solid #7f1d1d; border-radius: 10px;
    padding: 0.9rem 1.1rem; color: #fca5a5; margin-top: 0.6rem;
}
.error-card b {color: #fecaca;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """A friendly, user-facing error with a short title and detail."""
    def __init__(self, title: str, detail: str):
        self.title = title
        self.detail = detail
        super().__init__(f"{title}: {detail}")


def show_error(err: AgentError):
    st.markdown(
        f'<div class="error-card">⚠️ <b>{err.title}</b><br>{err.detail}</div>',
        unsafe_allow_html=True,
    )


def _headers():
    token = get_hf_token()

    if not token:
        raise AgentError(
            "Missing API token",
            "No Hugging Face token found. Put HF_TOKEN=hf_... in your .env file."
        )

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def call_hf_api(model_id: str, *, json_payload=None, data_payload=None,
                 max_retries: int = 3, wait_seconds: int = 35) -> requests.Response:
    """
    Shared HTTP layer for every tool. Centralizes error handling so each tool
    doesn't have to repeat it: network errors, timeouts, auth failures, rate
    limits, and the "model is still loading" 503 case (with automatic retry).
    """
    url = API_URL_TEMPLATE.format(model_id=model_id)
    headers = _headers()

    for attempt in range(1, max_retries + 1):
        try:
            if json_payload is not None:
                response = requests.post(url, headers=headers, json=json_payload, timeout=120)
            else:
                response = requests.post(url, headers=headers, data=data_payload, timeout=120)
        except requests.exceptions.Timeout:
            raise AgentError(
                "Request timed out",
                "Hugging Face didn't respond in time. This can happen with larger "
                "models - please try again.",
            )
        except requests.exceptions.ConnectionError:
            raise AgentError(
                "Network error",
                "Couldn't reach the Hugging Face API. Check your internet connection "
                "and try again.",
            )

        if response.status_code == 200:
            return response

        if response.status_code == 503:
            if attempt < max_retries:
                st.info(f"Model is warming up, retrying... ({attempt}/{max_retries})")
                time.sleep(wait_seconds)
                continue
            raise AgentError(
                "Model still loading",
                "The model didn't finish loading in time. Please try again in a moment.",
            )

        if response.status_code == 401:
            raise AgentError(
                "Invalid API token",
                "Your HF_API_TOKEN was rejected. Double-check it at "
                "huggingface.co/settings/tokens.",
            )

        if response.status_code == 429:
            raise AgentError(
                "Rate limit reached",
                "You've hit the free-tier request limit. Wait a bit before trying again.",
            )

        if response.status_code == 400:
            detail = _safe_error_detail(response)
            raise AgentError(
                "Request rejected",
                f"The model rejected this request - it may be invalid input. Details: {detail}",
            )

        detail = _safe_error_detail(response)
        raise AgentError(f"API error ({response.status_code})", detail)

    raise AgentError("Unexpected failure", "Ran out of retries without a clear error.")


def _safe_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("error", payload))
    except ValueError:
        return response.text[:300] if response.text else "No further details provided."


def validate_uploaded_image(uploaded_file) -> bytes:
    if uploaded_file is None:
        raise AgentError("No image", "No image was uploaded.")
    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise AgentError(
            "Image too large",
            f"That image is {size_mb:.1f} MB - please upload one under {MAX_UPLOAD_MB} MB.",
        )
    try:
        image_bytes = uploaded_file.getvalue()
        Image.open(io.BytesIO(image_bytes)).verify()
        return image_bytes
    except Exception:
        raise AgentError("Invalid image", "That file doesn't look like a valid image.")


# ---------------------------------------------------------------------------
# Tool 1: generate_image (text -> image)
# ---------------------------------------------------------------------------

def generate_image(prompt: str, model_id: str, negative_prompt: str = "") -> Image.Image:
    """Generate a new image from a text prompt using Hugging Face."""
    if not prompt.strip():
        raise AgentError(
            "Missing prompt",
            "Generating an image needs a text description."
        )

    # Use the single, shared Hugging Face client defined above.
    client = get_client()

    try:
        image = client.text_to_image(
            prompt=prompt,
            model=model_id,
            negative_prompt=negative_prompt.strip() or None,
        )
        return image
    except Exception as exc:
        raise AgentError(
            "Image generation failed",
            f"Hugging Face could not generate the image: {exc}"
        )
# ---------------------------------------------------------------------------
# Tool 2: describe_image (image -> text)
# ---------------------------------------------------------------------------

def describe_image(image_bytes: bytes) -> str:
    response = call_hf_api(CAPTION_MODEL, data_payload=image_bytes)
    try:
        result = response.json()
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        raise ValueError
    except Exception:
        raise AgentError("Unexpected response", "The captioning model returned something the app couldn't parse.")


# ---------------------------------------------------------------------------
# Tool 3: edit_image (image + instruction -> new image)
# ---------------------------------------------------------------------------

def edit_image(
    image_bytes: bytes,
    instruction: str,
    strength: float = 0.75
) -> Image.Image:

    if not instruction.strip():
        raise AgentError(
            "Missing instructions",
            "Tell the agent what change to make to the image."
        )

    try:
        client = get_client()

        # Send the uploaded image as the FIRST positional argument.
        result = client.image_to_image(
            image_bytes,
            prompt=instruction.strip(),
            model=EDIT_MODEL,
        )

        return result

    except Exception as exc:
        raise AgentError(
            "Image editing failed",
            f"Hugging Face could not edit the image: {exc}"
        )
# ---------------------------------------------------------------------------
# Router: decides which tool to use
# ---------------------------------------------------------------------------

def route_request(user_text: str, has_uploaded_image: bool):

    if not has_uploaded_image:
        return (
            "generate_image",
            "No image was uploaded, so I'll create a new image."
        )

    if not user_text.strip():
        return (
            "describe_image",
            "An image was uploaded with no instructions, so I'll describe it."
        )

    return (
        "edit_image",
        "An image was uploaded with instructions, so I'll edit it."
    )

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("🎨 Computer Vision Agent")
st.markdown(
    '<div class="agent-subtitle">Generate a new image, describe one, or recreate it with changes.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_label = st.selectbox("Image generation model", list(IMAGE_GEN_MODELS.keys()))
    model_id = IMAGE_GEN_MODELS[model_label]
    negative_prompt = st.text_area(
        "Negative prompt (optional, for new images)",
        placeholder="blurry, low quality, watermark, extra limbs...",
    )
    edit_strength = st.slider(
        "Edit strength (for recreating images)", min_value=0.1, max_value=2.0, value=0.75, step=0.05,
        help="Higher = the edited image follows your instruction more strongly and departs more from the original.",
    )
    st.markdown("---")
    st.markdown(
        "Get a free API token from "
        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
    )
    if not get_hf_token():
        st.warning("Hugging Face API token not found.")
    else:
        st.success("Hugging Face API token loaded.")

col_left, col_right = st.columns([3, 2])
with col_left:
    user_text = st.text_area(
        "What do you want?",
        placeholder="e.g. 'A serene mountain lake at sunrise' or 'Make the sky sunset orange'",
        height=140,
    )
with col_right:
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

run_clicked = st.button("✨ Run Agent", type="primary", use_container_width=True)

if run_clicked:
    try:
        if not user_text.strip() and not uploaded_file:
            raise AgentError("Nothing to do", "Please enter a request or upload an image.")

        image_bytes = validate_uploaded_image(uploaded_file) if uploaded_file else None
        tool_name, reason = route_request(user_text, uploaded_file is not None)

        st.markdown(f'<span class="tool-badge">🔧 {tool_name}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="reason-text">{reason}</div>', unsafe_allow_html=True)

        with st.spinner("Working on it..."):
            if tool_name == "generate_image":
                result_image = generate_image(user_text, model_id, negative_prompt)
                st.session_state["result"] = {"type": "image", "image": result_image, "caption": user_text}

            elif tool_name == "describe_image":
                description = describe_image(image_bytes)
                st.session_state["result"] = {
                    "type": "description", "image": Image.open(io.BytesIO(image_bytes)), "text": description,
                }

            elif tool_name == "edit_image":
                result_image = edit_image(image_bytes, user_text, edit_strength)
                st.session_state["result"] = {"type": "image", "image": result_image, "caption": f"Edited: {user_text}"}

    except AgentError as err:
        show_error(err)
        st.session_state.pop("result", None)

# --- Display result ---
result = st.session_state.get("result")
if result:
    if result["type"] == "image":
        st.image(result["image"], caption=result["caption"], use_container_width=True)
        buf = io.BytesIO()
        result["image"].save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            "⬇️ Download Image", data=buf, file_name="image_result.png",
            mime="image/png", use_container_width=True,
        )
    elif result["type"] == "description":
        st.image(result["image"], use_container_width=True)
        st.success(result["text"])
