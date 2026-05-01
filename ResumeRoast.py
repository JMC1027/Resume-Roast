# ─────────────────────────────────────────────
#  Resume Roast
#  Uploads a PDF resume, extracts the text, and
#  sends it to Claude for a scored, playful-but-
#  honest analysis. The UI is built with Gradio.
# ─────────────────────────────────────────────

import gradio as gr          # Gradio: builds the web UI
import anthropic             # Anthropic SDK: talks to Claude
import pdfplumber            # pdfplumber: extracts text from PDF files
from dotenv import load_dotenv  # python-dotenv: loads API key from .env file

# Load environment variables from the .env file so that
# ANTHROPIC_API_KEY is available without hardcoding it in source code.
load_dotenv()

# Create the Anthropic client. It automatically reads ANTHROPIC_API_KEY
# from the environment — no need to pass the key explicitly.
client = anthropic.Anthropic()

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
#  This is the instruction set sent to Claude
#  before every resume. It defines Claude's
#  persona, the exact output format we expect,
#  the scoring rubric, and the desired tone.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are Resume Roast — a brutally honest, secretly supportive career coach with the wit of a stand-up \
comedian and the eye of a senior hiring manager who has seen it all.

Analyze the resume and respond in this EXACT format (no deviation):

## SCORE: X/10

**The Roast:** [One punchy sentence that captures the biggest flaw — make it sting, but keep it true]

---

### The Good
[2-3 specific genuine strengths. No hollow compliments — cite actual content from the resume]

### The Bad
[2-3 real problems holding this person back. Be specific, not vague. "Weak action verbs" is vague; \
"'Assisted with' on every bullet tells me nothing about your actual impact" is specific]

### The Ugly
[The single most cringe-worthy thing on this resume — the thing that made you do a double-take]

### Glow-Up Tips
1. [Specific, actionable fix — not "improve your skills"]
2. [Specific, actionable fix]
3. [Specific, actionable fix]

---

Scoring guide (be stingy):
- 1-3: Needs a complete overhaul before it should be sent to anyone
- 4-5: Below average — a hiring manager skips this in 6 seconds
- 6-7: Gets the interview sometimes, but leaves points on the table
- 8-9: Strong resume with a couple of fixable issues
- 10: Reserved for near-perfection (if you give this out freely, you're lying)

Tone: You're the Gordon Ramsay of career coaching. Harsh but never cruel. \
Specific, never generic. You roast the resume, not the person.\
"""


def extract_text(file_path: str) -> str:
    """
    Opens a PDF file and extracts all readable text from every page.

    pdfplumber works on text-based PDFs (the kind exported from Word,
    Google Docs, etc.). Scanned image-only PDFs return empty strings
    because there is no embedded text layer to extract.

    Returns a single string with all pages joined by newlines,
    with leading/trailing whitespace removed.
    """
    with pdfplumber.open(file_path) as pdf:
        # Extract text from each page; fall back to "" if a page has no text
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def roast_resume(file):
    """
    Generator function that streams Claude's resume analysis back to the UI.

    Gradio supports generator functions: every time we `yield` a value,
    the UI updates in real time. This gives users live streaming output
    instead of waiting for the full response before seeing anything.

    Steps:
      1. Guard against no file being uploaded.
      2. Extract text from the PDF.
      3. Send the text to Claude with a streaming API call.
      4. Yield each text chunk as it arrives so the UI updates token-by-token.
    """

    # Step 1: Make sure the user actually uploaded something
    if file is None:
        yield "Upload a resume PDF above and hit the button to get roasted!"
        return

    # Gradio 4.x passes the file path as a plain string (type="filepath").
    # Older versions passed a file-like object with a .name attribute.
    # This handles both cases gracefully.
    file_path = file if isinstance(file, str) else file.name

    # Step 2: Pull the text out of the PDF
    try:
        resume_text = extract_text(file_path)
    except Exception as e:
        # Catches corrupt files, password-protected PDFs, etc.
        yield f"Couldn't read that PDF: {e}"
        return

    # If extraction succeeded but returned nothing, the PDF is likely a
    # scanned image with no embedded text layer — we can't process it.
    if not resume_text:
        yield "This PDF has no readable text. Scan-only PDFs aren't supported — export a text-based PDF and try again."
        return

    # Step 3 & 4: Stream the Claude response and yield each chunk to the UI
    result = ""  # Accumulates the full response as chunks arrive
    try:
        with client.messages.stream(
            model="claude-opus-4-7",   # Most capable Claude model
            max_tokens=1024,           # Enough for a thorough roast
            thinking={"type": "adaptive"},  # Lets Claude reason before responding
            system=SYSTEM_PROMPT,      # Persona + format instructions (sent once)
            messages=[{
                "role": "user",
                # The actual resume text is passed here as the user message
                "content": f"Here is the resume to roast:\n\n{resume_text}"
            }]
        ) as stream:
            # stream.text_stream yields only the visible text tokens,
            # filtering out internal "thinking" blocks automatically.
            for text in stream.text_stream:
                result += text   # Append new chunk to the running result
                yield result     # Push the updated result to the Gradio UI

    except anthropic.AuthenticationError:
        # Triggered when ANTHROPIC_API_KEY is missing or invalid
        yield "Invalid API key. Set `ANTHROPIC_API_KEY` in your `.env` file."
    except anthropic.APIError as e:
        # Catches all other API-level errors (rate limits, server errors, etc.)
        yield f"API error: {e}"


# ─────────────────────────────────────────────
#  GRADIO UI
#  gr.Blocks gives us full layout control.
#  The UI has two columns: upload + button on
#  the left, streaming output on the right.
# ─────────────────────────────────────────────
with gr.Blocks(title="Resume Roast") as app:

    # Page header
    gr.Markdown("""
# Resume Roast
### Upload your resume. Get the honest feedback no recruiter will ever give you.
""")

    with gr.Row():

        # Left column: file upload and trigger button
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Your Resume (PDF only)",
                file_types=[".pdf"],   # Restricts the file picker to PDFs only
                type="filepath"        # Tells Gradio to pass the path string, not file bytes
            )
            roast_btn = gr.Button("Roast My Resume", variant="primary", size="lg")
            gr.Markdown("_Your file is read locally and never stored._")

        # Right column: where the streaming roast output appears
        with gr.Column(scale=2):
            # gr.Markdown renders the response with formatting (bold, headers, etc.)
            output = gr.Markdown(value="_Your roast will appear here..._")

    # Wire the button click to roast_resume.
    # Because roast_resume is a generator, Gradio streams each yielded
    # value into `output` automatically.
    roast_btn.click(
        fn=roast_resume,
        inputs=file_input,
        outputs=output,
        show_progress="minimal"  # Shows a subtle spinner while waiting
    )

# Only launch the server when this file is run directly,
# not when it's imported as a module.
if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css="footer { display: none !important; }"  # Hides the Gradio branding footer
    )
