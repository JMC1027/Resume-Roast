
#  Resume Roast
#  Uploads a PDF resume, extracts the text,
#  sends it to Claude for a scored, playful-but-
#  honest analysis. The UI is built with Gradio.

import os
import gradio as gr
import anthropic
import pdfplumber
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env
load_dotenv()

# Anthropic client — reads ANTHROPIC_API_KEY from environment automatically
client = anthropic.Anthropic()


#  SYSTEM PROMPT
#  Defines Claude's persona, output format,
#  scoring rubric, and tone for every roast.

SYSTEM_PROMPT = """\
You are Resume Roast — a career coach who has reviewed 10,000 resumes and has lost all patience \
for mediocrity. You are funny, brutally direct, and occasionally savage, but everything you say \
is TRUE and USEFUL. Think: if a very smart, slightly mean best friend who works in HR reviewed \
your resume after two espressos.

Your goal: make them laugh, make them wince, and make them actually fix their resume.

Analyze the resume and respond in this EXACT format:

## SCORE: X/10

**Verdict:** [One sentence. Make it land. This should be quotable — the kind of thing they screenshot \
and send to their friends. No softening, no hedging.]

---

### What's Actually Working
[2-3 genuine strengths. Be specific — quote their resume back at them. If there's genuinely \
nothing good, say so. Don't invent strengths to be nice.]

### What's Killing Your Chances
[2-3 specific problems. Name the exact thing that's wrong. "Your summary reads like a LinkedIn \
template you filled in at 11pm" is useful. "Needs improvement" is not.]

### The Facepalm Moment
[The single thing that made you stop and stare. One specific, unambiguous call-out. \
Could be formatting, could be a claim, could be a typo. Make it hurt — lovingly.]

### Fix These First
1. [Concrete action — specific enough that they can do it today]
2. [Concrete action]
3. [Concrete action]

---

Scoring (be honest, not generous):
- 1–3: Do not send this to anyone. Seriously.
- 4–5: You'll get ignored by 9 out of 10 hiring managers
- 6–7: Gets a look sometimes. Leaves real opportunity on the table.
- 8–9: Solid. A few tweaks from genuinely impressive.
- 10: Basically perfect. You don't need this app.

Tone rules:
- Sarcasm is allowed. Cruelty is not.
- Every joke should have a point.
- Never pad. Never hedge. Never say "consider" when you mean "do this."
- You are roasting the resume. The person is trying their best. Keep that in mind.\
"""


def show_filename(file) -> str:
    if file is None:
        return ""
    return f"📄 {os.path.basename(file if isinstance(file, str) else file.name)}"


def extract_text(file_path: str) -> str:
    """
    Extracts all readable text from a PDF, page by page.
    Returns empty string for scanned/image-only PDFs.
    """
    with pdfplumber.open(file_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def roast_resume(file):
    """
    Generator that streams Claude's roast back to the Gradio UI token by token.
    Each yield updates the output in real time.
    """

    if file is None:
        yield "Drop a PDF above and brace yourself."
        return

    # Gradio 4.x passes filepath as string; older versions used a file object
    file_path = file if isinstance(file, str) else file.name

    try:
        resume_text = extract_text(file_path)
    except Exception as e:
        yield f"Couldn't read that PDF: {e}"
        return

    if not resume_text:
        yield "This PDF has no readable text — it's probably a scanned image. Export a real PDF and try again."
        return

    result = ""
    try:
        with client.messages.stream(
            model="claude-opus-4-7",          # Most capable Claude model
            max_tokens=1024,                  # Enough for a thorough roast
            thinking={"type": "adaptive"},    # Claude reasons before responding
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Here is the resume:\n\n{resume_text}"
            }]
        ) as stream:
            # text_stream skips thinking blocks — only yields visible output
            for text in stream.text_stream:
                result += text
                yield result

    except anthropic.AuthenticationError:
        yield "Invalid API key. Set `ANTHROPIC_API_KEY` in your `.env` file."
    except anthropic.APIError as e:
        yield f"API error: {e}"


# ─────────────────────────────────────────────
#  CUSTOM CSS
#  Styles the page beyond what Gradio's theme
#  provides — fonts, colors, layout tweaks,
#  button animation, and output card styling.
# ─────────────────────────────────────────────
CSS = """
/* Import a bold display font from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500&display=swap');

/* Root page */
body, .gradio-container {
    background: #0f0f0f !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Big headline */
#headline h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ff6b35, #ff4500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
}

#headline p, #headline h3 {
    color: #bbb !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
}

/* Upload button (gr.UploadButton) */
#file-upload,
#file-upload label,
#file-upload button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 56px !important;
    background: #1a1a1a !important;
    border: 2px solid #ff6b35 !important;
    border-radius: 10px !important;
    color: #ff6b35 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(255, 107, 53, 0.2) !important;
}
#file-upload:hover,
#file-upload label:hover,
#file-upload button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255, 107, 53, 0.4) !important;
}

/* Uploaded filename confirmation */
#file-name { min-height: 0 !important; }
#file-name p {
    color: #aaa !important;
    font-size: 0.82rem !important;
    text-align: center !important;
    margin: 2px 0 6px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* The roast button */
#roast-btn {
    background: linear-gradient(135deg, #ff6b35, #ff4500) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 14px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(255, 107, 53, 0.35) !important;
}
#roast-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255, 107, 53, 0.55) !important;
}
#roast-btn:active {
    transform: translateY(0px) !important;
}

/* Output card */
#output-card {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    min-height: 200px !important;
}

/* Markdown inside output card */
#output-card, #output-card * { color: #e8e8e8 !important; }
#output-card h2 { color: #ff6b35 !important; font-family: 'Syne', sans-serif !important; }
#output-card h3 { color: #ffffff !important; border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; }
#output-card strong { color: #ff6b35 !important; }
#output-card p, #output-card li { color: #e8e8e8 !important; line-height: 1.7 !important; }
#output-card hr { border-color: #2a2a2a !important; }
#output-card code {
    background: transparent !important;
    color: inherit !important;
    font-family: 'Inter', sans-serif !important;
    font-size: inherit !important;
    padding: 0 !important;
    border: none !important;
}

/* Disclaimer text */
#disclaimer { color: #888 !important; font-size: 0.8rem !important; }


/* Hide Gradio footer */
footer { display: none !important; }
"""

# ─────────────────────────────────────────────
#  JAVASCRIPT
#  Runs on page load:
#  - Adds a 🔥 favicon dynamically
#  - Animates the score when it appears in output
# ─────────────────────────────────────────────
JS = """
() => {
    // Set a fire emoji as the browser tab favicon
    const link = document.createElement('link');
    link.rel = 'icon';
    link.href = 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔥</text></svg>';
    document.head.appendChild(link);

    // Watch DOM for score animation trigger
    const observer = new MutationObserver(() => {

        // --- Score pulse animation ---
        const scoreEl = document.querySelector('#output-card h2');
        if (scoreEl && !scoreEl.dataset.animated) {
            scoreEl.dataset.animated = 'true';
            scoreEl.style.transition = 'transform 0.3s ease, opacity 0.3s ease';
            scoreEl.style.transform = 'scale(1.08)';
            scoreEl.style.opacity = '0.7';
            setTimeout(() => {
                scoreEl.style.transform = 'scale(1)';
                scoreEl.style.opacity = '1';
            }, 350);
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}
"""

# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(title="Resume Roast") as app:

    # Header
    gr.Markdown("""
# Resume Roast
### Your resume. Judged. Mercilessly.
""", elem_id="headline")

    with gr.Row():

        # Left: upload + button
        with gr.Column(scale=1, elem_classes=["upload-container"]):
            # gr.State holds the uploaded filepath between upload and roast click
            file_state = gr.State(None)
            # UploadButton always opens the file picker on every click (unlike gr.File)
            file_input = gr.UploadButton(
                "Upload Resume (PDF)",
                file_types=[".pdf"],
                type="filepath",
                elem_id="file-upload"
            )
            file_name = gr.Markdown("", elem_id="file-name")
            roast_btn = gr.Button(
                "🔥 Roast My Resume",
                variant="primary",
                size="lg",
                elem_id="roast-btn"
            )
            gr.Markdown("_Read locally. Never stored. No mercy._", elem_id="disclaimer")

        # Right: streaming output
        with gr.Column(scale=2):
            output = gr.Markdown(
                value="_Your resume roast will appear here..._",
                elem_id="output-card"
            )

    # On upload: store filepath in state and show filename
    file_input.upload(
        fn=lambda f: (f, show_filename(f)),
        inputs=file_input,
        outputs=[file_state, file_name]
    )

    roast_btn.click(
        fn=roast_resume,
        inputs=file_state,
        outputs=output,
        show_progress="minimal"
    )

if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="zinc"),
        css=CSS,
        js=JS
    )
