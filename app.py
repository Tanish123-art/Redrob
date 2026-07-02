import csv
import json
import os
import subprocess
import sys
import tempfile

import gradio as gr

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_candidates.json")
RANK_SCRIPT  = os.path.join(os.path.dirname(__file__), "rank.py")
TEMP_DIR     = tempfile.gettempdir()


def load_sample():
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def to_jsonl(raw_bytes, is_gz):
    if is_gz:
        return raw_bytes
    try:
        text = raw_bytes.decode("utf-8").strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return "\n".join(json.dumps(item) for item in parsed).encode("utf-8")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return raw_bytes


def run_ranking(upload_file, json_text):
    # determine input source
    if upload_file is not None:
        src = upload_file.name if hasattr(upload_file, "name") else upload_file
        is_gz = src.endswith(".gz")
        with open(src, "rb") as f:
            raw = f.read()
        raw = to_jsonl(raw, is_gz)
        suffix = ".jsonl.gz" if is_gz else ".jsonl"
    elif json_text and json_text.strip():
        text = json_text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                text = "\n".join(json.dumps(item) for item in parsed)
        except json.JSONDecodeError:
            pass
        raw = text.encode("utf-8")
        suffix = ".jsonl"
    else:
        return [["error", "", "", "no input provided"]], None

    # write input to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(prefix="redrob_ranking_", suffix=".csv", delete=False) as out_tmp:
        out_path = out_tmp.name

    # run ranker (challenge cap is 5 min on CPU; leave a safety margin)
    try:
        proc = subprocess.run(
            [sys.executable, RANK_SCRIPT, "--candidates", tmp_path, "--out", out_path],
            capture_output=True, text=True, timeout=290
        )
        if proc.returncode != 0:
            return [["error", "", "", proc.stderr or proc.stdout]], None
    except subprocess.TimeoutExpired:
        return [["error", "", "", "ranking timed out (>290s)"]], None
    except Exception as e:
        return [["error", "", "", str(e)]], None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not os.path.exists(out_path):
        return [["error", "", "", "output file not produced"]], None

    # parse CSV into table rows
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append([row["rank"], row["candidate_id"], row["score"], row["reasoning"]])

    if not rows:
        return [["info", "", "", "no candidates passed the filters"]], None

    return rows, out_path


with gr.Blocks(title="Redrob Ranking Engine") as demo:
    gr.Markdown("**Redrob Candidate Ranking Engine** — Team: The defenders | Solo: Tanish M")

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="candidates file (.jsonl / .jsonl.gz / .json)")
            json_input = gr.Textbox(
                label="or paste JSON here",
                lines=8,
                value=load_sample()
            )
            run_btn = gr.Button("Run Ranking")
            download_btn = gr.DownloadButton("Download results.csv", visible=False)

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                headers=["Rank", "Candidate ID", "Score", "Reasoning"],
                label="Results",
                wrap=True,
                interactive=False
            )

    def run_ranking_ui(upload_file, json_text):
        rows, out_path = run_ranking(upload_file, json_text)
        return rows, gr.DownloadButton(visible=out_path is not None, value=out_path)

    run_btn.click(
        fn=run_ranking_ui,
        inputs=[upload, json_input],
        outputs=[output_table, download_btn]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        ssr_mode=False,
    )