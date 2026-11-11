import os
import re
import uuid
import json
import threading
import subprocess
import sys

from pathlib import Path
from datetime import datetime

import cv2

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    abort,
    render_template_string
)

from werkzeug.utils import secure_filename


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "web_outputs"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

PIPELINE = ROOT / "pipeline.py"

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024

ALLOWED = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm"
}

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

jobs = {}
lock = threading.Lock()


# ============================================================
# FRONTEND
# ============================================================

PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>RoadScene | Road Intelligence</title>

<link rel="preconnect" href="https://fonts.googleapis.com">

<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap"
rel="stylesheet">


<style>

:root {
    --navy: #142a43;
    --blue: #3976ee;
    --muted: #7b8998;
    --line: #e5eaf0;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #f3f6f9;
    color: #1d2b3a;
    font-family: 'DM Sans', sans-serif;
}

.layout {
    display: flex;
    min-height: 100vh;
}

.side {
    width: 220px;
    background: #12253b;
    color: white;
    padding: 27px 18px;
    display: flex;
    flex-direction: column;
}

.brand {
    font: 700 18px 'Space Grotesk';
    margin-bottom: 55px;
}

.brand i {
    font-style: normal;
    background: #3976ee;
    padding: 9px;
    border-radius: 9px;
    margin-right: 9px;
}

.side small {
    color: #8094aa;
    font-size: 9px;
    letter-spacing: 1.5px;
}

.side .nav {
    background: #203d5d;
    padding: 13px;
    border-radius: 8px;
    margin-top: 12px;
    font-size: 12px;
}

.side .foot {
    margin-top: auto;
    color: #9db0c3;
    font-size: 10px;
    border-top: 1px solid #294057;
    padding-top: 15px;
}

.main {
    width: min(1350px, 100%);
    margin: auto;
    padding: 30px 4%;
}

.top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.eyebrow {
    font-size: 9px;
    letter-spacing: 1.7px;
    color: #8292a3;
    font-weight: 700;
}

.top h1 {
    font: 700 25px 'Space Grotesk';
    margin: 5px 0;
}

.top a {
    font-size: 11px;
    color: #42658a;
    text-decoration: none;
    background: white;
    border: 1px solid var(--line);
    padding: 10px;
    border-radius: 7px;
}

.hero {
    background: #172f4b;
    color: white;
    border-radius: 13px;
    padding: 26px 30px;
    margin: 25px 0 18px;
}

.hero .eyebrow {
    color: #a9c2dd;
}

.hero h2 {
    font: 700 29px/1.12 'Space Grotesk';
    margin: 12px 0;
}

.hero em {
    color: #78a9ff;
    font-style: normal;
}

.hero p {
    font-size: 11px;
    color: #c3cfdb;
    max-width: 550px;
    line-height: 1.7;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 17px;
}

.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 20px;
    box-shadow: 0 4px 15px #1b355008;
}

.heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.heading h3 {
    font: 600 14px 'Space Grotesk';
    margin: 0;
}

.step {
    color: #3976ee;
    background: #edf3ff;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 9px;
    margin-right: 8px;
}

.drop {
    height: 150px;
    border: 1px dashed #b8c8d9;
    border-radius: 9px;
    background: #f9fbfd;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    cursor: pointer;
    text-align: center;
}

.drop.drag {
    background: #edf4ff;
    border-color: #3976ee;
}

.drop strong {
    font-size: 12px;
}

.drop span {
    font-size: 10px;
    color: #75869a;
    margin-top: 7px;
}

.drop small {
    font-size: 8px;
    color: #9ba7b3;
    margin-top: 10px;
}

.drop input {
    display: none;
}

.file {
    background: #eff5ff;
    padding: 10px;
    border-radius: 7px;
    margin-top: 10px;
    font-size: 10px;
    display: flex;
    justify-content: space-between;
}

.settings {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 18px;
}

.settings label {
    font-size: 9px;
    color: #718092;
}

.settings input,
.settings select {
    display: block;
    width: 100%;
    padding: 9px;
    border: 1px solid #dfe5ec;
    border-radius: 6px;
    margin-top: 5px;
    background: white;
}

.hint {
    font-size: 9px;
    color: #98a3af;
}

.btn {
    width: 100%;
    border: 0;
    border-radius: 7px;
    background: #3475ef;
    color: white;
    padding: 13px;
    margin-top: 13px;
    font-weight: 700;
    cursor: pointer;
}

.btn:disabled {
    opacity: .45;
    cursor: not-allowed;
}

.status {
    padding: 14px;
    background: #f6f8fb;
    border-radius: 8px;
    font-size: 11px;
    margin-bottom: 20px;
}

.status p {
    color: #8593a2;
    font-size: 10px;
}

.bar {
    height: 7px;
    background: #edf1f5;
    border-radius: 9px;
    overflow: hidden;
    margin: 9px 0;
}

.bar div {
    height: 100%;
    width: 0;
    background: #3976ee;
    transition: width .3s;
}

.meta {
    display: flex;
    justify-content: space-between;
    color: #8290a0;
    font-size: 9px;
}

.models {
    margin-top: 22px;
    border-top: 1px solid #edf0f3;
}

.model {
    padding: 12px 0;
    border-bottom: 1px solid #f0f2f5;
    font-size: 10px;
}

.model b {
    display: block;
}

.model span {
    color: #94a0ac;
    font-size: 9px;
}

.error {
    background: #fff0f0;
    color: #a33;
    padding: 10px;
    border-radius: 6px;
    font-size: 10px;
    margin-top: 10px;
}

.results {
    margin-top: 25px;
}

.resultgrid {
    display: grid;
    grid-template-columns: 1.4fr .8fr;
    gap: 16px;
}

.results video {
    width: 100%;
    max-height: 450px;
    background: #101820;
    border-radius: 7px;
}

.download {
    color: #3976ee;
    font-size: 10px;
    text-decoration: none;
}

.metric {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #eee;
    padding: 12px 0;
    font-size: 10px;
    color: #758395;
}

.metric b {
    font: 700 17px 'Space Grotesk';
    color: #1c2e43;
}

.risks {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 7px;
    margin: 15px 0;
}

.risk {
    text-align: center;
    padding: 10px 3px;
    border-radius: 7px;
}

.risk b {
    display: block;
    font-size: 18px;
}

.risk span {
    font-size: 8px;
}

.high {
    background: #fff0f0;
    color: #d65353;
}

.med {
    background: #fff5e8;
    color: #d48b2c;
}

.low {
    background: #eaf8f2;
    color: #1c9a70;
}

.classrow {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #eee;
}

footer {
    color: #8996a3;
    font-size: 9px;
    margin-top: 30px;
    border-top: 1px solid var(--line);
    padding-top: 15px;
}

@media(max-width:850px) {

    .side {
        width: 65px;
        padding: 18px 9px;
    }

    .brand {
        font-size: 0;
    }

    .brand i {
        font-size: 14px;
    }

    .side small,
    .side .nav,
    .side .foot {
        display: none;
    }

    .grid,
    .resultgrid {
        grid-template-columns: 1fr;
    }

    .main {
        padding: 22px 13px;
    }
}

</style>

</head>


<body>

<div class="layout">

<aside class="side">

<div class="brand">
<i>RS</i>RoadScene
</div>

<small>WORKSPACE</small>

<div class="nav">
▣ &nbsp; Video Analysis
</div>

<div class="foot">
● LOCAL INFERENCE SERVER
<br><br>
Road_Scene_Know
</div>

</aside>


<main class="main">

<header class="top">

<div>

<div class="eyebrow">
COMPUTER VISION / INDIAN ROAD CONDITIONS
</div>

<h1>Road Scene Analysis</h1>

</div>

<a target="_blank"
href="https://github.com/Arghya-Chakraborty0812/Road_Scene_Know">

View GitHub ↗

</a>

</header>


<section class="hero">

<div class="eyebrow">
CAMERA-ONLY PERCEPTION
</div>

<h2>
Understand the road.
<br>
<em>Anticipate the risk.</em>
</h2>

<p>
Upload a driving video to detect and track road users,
map drivable space, estimate depth and visualize potential risk.
</p>

</section>


<div class="grid">


<!-- UPLOAD CARD -->

<section class="card">

<div class="heading">

<h3>
<span class="step">01</span>
Upload video
</h3>

<small>MP4 · MOV · AVI · MKV · WEBM</small>

</div>


<label class="drop" id="drop">

<input id="video"
type="file"
accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm">

<strong>＋ Drop your driving video here</strong>

<span>or click to browse files</span>

<small>Maximum upload size: 1 GB</small>

</label>


<div class="file" id="fileInfo" hidden>

<span id="fileName"></span>

<span id="fileSize"></span>

</div>


<!-- UPLOAD PROGRESS -->

<div id="uploadBox" hidden>

<p class="hint" id="uploadMessage">
Uploading video...
</p>

<div class="bar">
<div id="uploadBar"></div>
</div>

<div class="meta">

<span>UPLOAD PROGRESS</span>

<b id="uploadPct">0%</b>

</div>

</div>


<div class="heading" style="margin:20px 0 10px">

<h3>
<span class="step">02</span>
Inference settings
</h3>

<small>OPTIONAL</small>

</div>


<div class="settings">

<label>
Detector image size

<select id="imgsz">

<option>640</option>
<option>512</option>
<option>768</option>
<option>960</option>

</select>

</label>


<label>
Confidence threshold

<input id="conf"
type="number"
min=".01"
max=".99"
step=".05"
value=".25">

</label>


<label>
Segmentation refresh

<input id="seg"
type="number"
min="1"
max="30"
value="3">

</label>


<label>
Depth refresh

<input id="depth"
type="number"
min="1"
max="30"
value="2">

</label>

</div>


<p class="hint">
Lower refresh intervals provide more frequent estimates but take longer.
</p>


<button class="btn" id="run" disabled>
Run road analysis →
</button>


<div id="error" class="error" hidden></div>

</section>



<!-- STATUS CARD -->

<section class="card">

<div class="heading">

<h3>
<span class="step">03</span>
Processing status
</h3>

<small style="color:#229b75">
● LIVE
</small>

</div>


<div class="status">

<b id="jobTitle">Ready for a video</b>

<p id="message">
Your analysis will appear here once processing begins.
</p>

</div>


<div class="meta">

<span>INFERENCE PROGRESS</span>

<b id="pct">0%</b>

</div>


<div class="bar">
<div id="bar"></div>
</div>


<div class="meta">

<span id="frames">0 / 0 frames</span>

<span id="state">IDLE</span>

</div>


<div class="models">

<div class="model">

<b>01 · YOLO11s + ByteTrack</b>

<span>Object detection and persistent tracking</span>

</div>


<div class="model">

<b>02 · SegFormer</b>

<span>Drivable-area semantic segmentation</span>

</div>


<div class="model">

<b>03 · Depth Anything V2</b>

<span>Monocular depth estimation</span>

</div>


<div class="model">

<b>04 · Risk fusion</b>

<span>Distance, TTC and ego-path context</span>

</div>

</div>

</section>

</div>



<!-- RESULTS -->

<section class="results" id="results" hidden>

<div class="heading">

<div>

<div class="eyebrow">ANALYSIS OUTPUT</div>

<h2 style="font-family:'Space Grotesk'">
Video results
</h2>

</div>

<b style="font-size:9px;color:#16835e">
✓ ANALYSIS COMPLETE
</b>

</div>


<div class="resultgrid">


<div class="card">

<div class="heading">

<h3>Processed video</h3>

<a class="download" id="videoDownload">
Download MP4 ↓
</a>

</div>

<video id="player" controls playsinline></video>

</div>



<div class="card">

<div class="heading">

<h3>Detection summary</h3>

<a class="download" id="jsonDownload">
JSONL ↓
</a>

</div>


<div class="metric">
Frames analyzed
<b id="mFrames">—</b>
</div>


<div class="metric">
Object observations
<b id="mObjects">—</b>
</div>


<div class="risks">

<div class="risk high">
<b id="high">0</b>
<span>HIGH</span>
</div>

<div class="risk med">
<b id="medium">0</b>
<span>MED</span>
</div>

<div class="risk low">
<b id="low">0</b>
<span>LOW</span>
</div>

</div>


<small class="eyebrow">
MOST OBSERVED CLASSES
</small>

<div id="classes"></div>

</div>

</div>


<p class="hint">
Research prototype only. Monocular distance estimates and heuristic
risk levels are not a certified collision-warning system.
</p>

</section>


<footer>
ROAD_SCENE_KNOW · Built for research and Indian driving conditions
</footer>


</main>

</div>



<script>

// ============================================================
// ELEMENTS
// ============================================================

const $ = id => document.getElementById(id);

let chosen = null;
let timer = null;
let uploadRequest = null;


// ============================================================
// ERROR HANDLING
// ============================================================

function err(message) {

    $('error').textContent = message;

    $('error').hidden = false;

}


// ============================================================
// FILE VALIDATION
// ============================================================

function choose(file) {

    $('error').hidden = true;

    if (!file) return;


    // FIXED EXTENSION VALIDATION

    if (!/\.(mp4|mov|avi|mkv|webm)$/i.test(file.name)) {

        return err(
            'Choose MP4, MOV, AVI, MKV or WEBM.'
        );

    }


    if (file.size === 0) {

        return err('The selected video is empty.');

    }


    if (file.size > 1024 ** 3) {

        return err('File exceeds 1 GB.');

    }


    chosen = file;


    $('fileInfo').hidden = false;

    $('fileName').textContent = file.name;

    $('fileSize').textContent =
        (file.size / 1048576).toFixed(2) + ' MB';


    $('run').disabled = false;

    $('run').textContent = 'Run road analysis →';

}


// ============================================================
// FILE INPUT
// ============================================================

$('video').addEventListener('change', e => {

    choose(e.target.files[0]);

});


// ============================================================
// DRAG AND DROP
// ============================================================

$('drop').addEventListener('dragover', e => {

    e.preventDefault();

    $('drop').classList.add('drag');

});


$('drop').addEventListener('dragleave', () => {

    $('drop').classList.remove('drag');

});


$('drop').addEventListener('drop', e => {

    e.preventDefault();

    $('drop').classList.remove('drag');

    choose(e.dataTransfer.files[0]);

});


// ============================================================
// BACKEND HEALTH
// ============================================================

fetch('/api/health')

.then(r => r.json())

.then(d => {

    if (!d.ok) {

        err(
            'Missing project files: ' + d.missing.join(', ')
        );

    }

})

.catch(() => {

    err('Backend is not responding.');

});


// ============================================================
// START ANALYSIS
// ============================================================

$('run').onclick = function() {

    if (!chosen) return;


    $('error').hidden = true;

    $('run').disabled = true;

    $('run').textContent = 'Uploading video...';


    $('uploadBox').hidden = false;

    $('uploadBar').style.width = '0%';

    $('uploadPct').textContent = '0%';

    $('uploadMessage').textContent = 'Uploading video...';


    const form = new FormData();

    form.append('video', chosen);

    form.append('imgsz', $('imgsz').value);

    form.append('conf', $('conf').value);

    form.append('seg_every', $('seg').value);

    form.append('depth_every', $('depth').value);


    // XHR PROVIDES REAL UPLOAD PROGRESS

    const xhr = new XMLHttpRequest();

    uploadRequest = xhr;


    xhr.open('POST', '/api/analyze', true);


    xhr.upload.onprogress = function(e) {

        if (e.lengthComputable) {

            const percent = Math.round(
                (e.loaded / e.total) * 100
            );

            $('uploadBar').style.width = percent + '%';

            $('uploadPct').textContent = percent + '%';

            $('uploadMessage').textContent =
                'Uploading video: ' + percent + '%';

        }

    };


    xhr.onload = function() {

        let data;

        try {

            data = JSON.parse(xhr.responseText);

        }

        catch {

            err(
                'Server returned an invalid response. HTTP ' +
                xhr.status
            );

            resetButton();

            return;

        }


        if (xhr.status < 200 || xhr.status >= 300) {

            err(data.error || 'Upload failed.');

            resetButton();

            return;

        }


        $('uploadMessage').textContent =
            'Upload completed. Starting inference...';

        $('uploadBar').style.width = '100%';

        $('uploadPct').textContent = '100%';


        $('jobTitle').textContent = 'Analysis in progress';

        $('message').textContent =
            'Models are processing your video.';

        $('state').textContent = 'RUNNING';

        $('results').hidden = true;


        if (timer) clearInterval(timer);


        poll(data.job_id);

        timer = setInterval(
            () => poll(data.job_id),
            1800
        );

    };


    xhr.onerror = function() {

        err(
            'Network error. Check whether Flask is running.'
        );

        resetButton();

    };


    xhr.onabort = function() {

        err('Upload cancelled.');

        resetButton();

    };


    xhr.ontimeout = function() {

        err('Upload timed out.');

        resetButton();

    };


    xhr.timeout = 0;

    xhr.send(form);

};


// ============================================================
// RESET BUTTON
// ============================================================

function resetButton() {

    $('run').disabled = false;

    $('run').textContent = 'Try again →';

}


// ============================================================
// POLL JOB STATUS
// ============================================================

async function poll(id) {

    try {

        const response = await fetch(
            '/api/status/' + encodeURIComponent(id)
        );


        if (!response.ok) {

            throw new Error(
                'Status request failed: HTTP ' + response.status
            );

        }


        const d = await response.json();


        $('bar').style.width = d.progress + '%';

        $('pct').textContent = d.progress + '%';


        $('frames').textContent =
            (d.processed_frames || 0) +
            ' / ' +
            (d.total_frames || 0) +
            ' frames';


        $('message').textContent = d.message;

        $('state').textContent = d.state.toUpperCase();


        // ====================================================
        // COMPLETED
        // ====================================================

        if (d.state === 'completed') {

            clearInterval(timer);

            timer = null;


            $('jobTitle').textContent =
                'Your video is ready';


            $('results').hidden = false;


            $('player').src = d.output_url;

            $('player').load();


            $('videoDownload').href = d.output_url;

            $('videoDownload').download =
                'road_scene_output.mp4';


            $('jsonDownload').href = d.json_url || '#';

            $('jsonDownload').download =
                'road_scene_output.jsonl';


            const s = d.summary;


            $('mFrames').textContent =
                s.frames.toLocaleString();


            $('mObjects').textContent =
                s.objects.toLocaleString();


            $('high').textContent =
                s.risk.HIGH || 0;


            $('medium').textContent =
                s.risk.MED || 0;


            $('low').textContent =
                s.risk.LOW || 0;


            const arr = Object.entries(s.classes)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8);


            $('classes').replaceChildren();


            arr.forEach(([name, count]) => {

                const row = document.createElement('div');

                row.className = 'classrow';


                const label = document.createElement('span');

                label.textContent = name;


                const value = document.createElement('b');

                value.textContent = count;


                row.append(label, value);

                $('classes').appendChild(row);

            });


            if (!arr.length) {

                $('classes').textContent =
                    'No objects detected.';

            }


            $('run').disabled = false;

            $('run').textContent =
                'Analyze another video →';


            $('results').scrollIntoView({
                behavior: 'smooth'
            });

        }


        // ====================================================
        // FAILED
        // ====================================================

        else if (d.state === 'failed') {

            clearInterval(timer);

            timer = null;


            err(d.message);

            $('jobTitle').textContent =
                'Analysis failed';


            resetButton();

        }

    }

    catch(e) {

        clearInterval(timer);

        timer = null;

        err(e.message);

        resetButton();

    }

}

</script>

</body>
</html>
"""


# ============================================================
# CHECK REQUIRED ASSETS
# ============================================================

def missing_assets():

    missing = []

    required = [
        PIPELINE,
        ROOT / "best-2.pt",
        ROOT / "segformer_idd"
    ]

    for p in required:

        if not p.exists():

            missing.append(
                p.name + ("/" if p.is_dir() else "")
            )

    return missing


# ============================================================
# WORKER PROCESS
# ============================================================

def worker(jid, src, out_name, cfg):

    out = OUTPUTS / out_name

    logpath = out.with_suffix(".jsonl")


    try:

        # ----------------------------------------------------
        # GET VIDEO INFORMATION
        # ----------------------------------------------------

        cap = cv2.VideoCapture(str(src))

        if not cap.isOpened():

            raise RuntimeError(
                "Unable to open uploaded video. "
                "Check video format or codec."
            )


        total = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )

        cap.release()


        with lock:

            jobs[jid]["total_frames"] = total

            jobs[jid]["state"] = "running"


        # ----------------------------------------------------
        # PIPELINE COMMAND
        # ----------------------------------------------------

        cmd = [

            sys.executable,

            str(PIPELINE),

            "--src", str(src),

            "--out", str(out),

            "--weights", str(ROOT / "best-2.pt"),

            "--imgsz", str(cfg["imgsz"]),

            "--seg_every", str(cfg["seg_every"]),

            "--depth_every", str(cfg["depth_every"]),

            "--conf", str(cfg["conf"])

        ]


        print("Starting pipeline:", " ".join(cmd))


        # ----------------------------------------------------
        # EXECUTE PIPELINE
        # ----------------------------------------------------

        proc = subprocess.Popen(

            cmd,

            cwd=ROOT,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1

        )


        tail = []


        for line in proc.stdout:

            line = line.rstrip()

            if not line:

                continue


            print(line)


            tail = (tail + [line])[-25:]


            # FIXED REGEX

            m = re.search(
                r"frame\s+(\d+)",
                line,
                re.I
            )


            with lock:

                jobs[jid]["log"] = tail


                if m:

                    n = int(m.group(1))

                    jobs[jid]["processed_frames"] = n


                    if total:

                        jobs[jid]["progress"] = min(
                            99,
                            int(n * 100 / total)
                        )


                    jobs[jid]["message"] = (
                        f"Processed {n:,} frames"
                    )


        code = proc.wait()


        if code != 0:

            raise RuntimeError(
                "Pipeline exited with code "
                + str(code)
                + ". Check model files and dependencies. "
                + "Last logs: "
                + " | ".join(tail[-5:])
            )


        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if not out.exists() or out.stat().st_size == 0:

            raise RuntimeError(
                "Output video was not generated."
            )


        # ----------------------------------------------------
        # GENERATE SUMMARY
        # ----------------------------------------------------

        summary = {

            "frames": 0,

            "objects": 0,

            "risk": {

                "HIGH": 0,

                "MED": 0,

                "LOW": 0

            },

            "classes": {}

        }


        if logpath.exists():

            with logpath.open(
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    try:

                        rec = json.loads(line)

                    except (json.JSONDecodeError, TypeError):

                        continue


                    summary["frames"] += 1


                    for ob in rec.get("objects", []):

                        summary["objects"] += 1


                        risk = ob.get(
                            "risk",
                            "LOW"
                        )

                        if risk not in summary["risk"]:

                            risk = "LOW"


                        summary["risk"][risk] += 1


                        cls = str(
                            ob.get("cls", "unknown")
                        )


                        summary["classes"][cls] = (
                            summary["classes"].get(cls, 0) + 1
                        )


        # ----------------------------------------------------
        # JOB COMPLETED
        # ----------------------------------------------------

        with lock:

            jobs[jid].update(

                state="completed",

                progress=100,

                message="Analysis completed.",

                output_url="/outputs/" + out.name,

                json_url=(
                    "/outputs/" + logpath.name
                    if logpath.exists()
                    else None
                ),

                summary=summary

            )


        print("Job completed:", jid)


    except Exception as e:

        print("Worker error:", str(e))


        with lock:

            jobs[jid].update(

                state="failed",

                message=str(e)

            )


    finally:

        # Remove original upload after processing.
        # The generated output remains available.

        try:

            if src.exists():

                src.unlink()

        except OSError:

            pass


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return render_template_string(PAGE)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    m = missing_assets()

    return jsonify(

        ok=not m,

        missing=m

    )


# ============================================================
# VIDEO UPLOAD + ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze():

    # --------------------------------------------------------
    # CHECK ASSETS
    # --------------------------------------------------------

    m = missing_assets()

    if m:

        return jsonify(

            error=(
                "Missing required files in the same folder "
                "as app.py: " + ", ".join(m)
            )

        ), 400


    # --------------------------------------------------------
    # GET VIDEO
    # --------------------------------------------------------

    f = request.files.get("video")


    if not f or not f.filename:

        return jsonify(
            error="Choose a video first."
        ), 400


    filename = secure_filename(f.filename)


    if not filename:

        return jsonify(
            error="Invalid filename."
        ), 400


    ext = Path(filename).suffix.lower()


    if ext not in ALLOWED:

        return jsonify(
            error="Unsupported video format."
        ), 400


    # --------------------------------------------------------
    # VALIDATE SETTINGS
    # --------------------------------------------------------

    try:

        cfg = {

            "imgsz": int(
                request.form.get("imgsz", 640)
            ),

            "seg_every": int(
                request.form.get("seg_every", 3)
            ),

            "depth_every": int(
                request.form.get("depth_every", 2)
            ),

            "conf": float(
                request.form.get("conf", .25)
            )

        }


        if not 320 <= cfg["imgsz"] <= 1280:

            raise ValueError()


        if not 1 <= cfg["seg_every"] <= 30:

            raise ValueError()


        if not 1 <= cfg["depth_every"] <= 30:

            raise ValueError()


        if not .01 <= cfg["conf"] <= .99:

            raise ValueError()


    except (ValueError, TypeError):

        return jsonify(

            error=(
                "Invalid settings. Check image size, "
                "refresh intervals and confidence."
            )

        ), 400


    # --------------------------------------------------------
    # SAVE UPLOAD
    # --------------------------------------------------------

    jid = uuid.uuid4().hex[:12]


    src = UPLOADS / (jid + ext)


    out_name = jid + "_analyzed.mp4"


    try:

        f.save(src)

    except Exception as e:

        return jsonify(

            error="Unable to save uploaded video: " + str(e)

        ), 500


    if not src.exists() or src.stat().st_size == 0:

        return jsonify(

            error="Uploaded file is empty or could not be saved."

        ), 400


    # --------------------------------------------------------
    # INITIALIZE JOB
    # --------------------------------------------------------

    with lock:

        jobs[jid] = {

            "state": "queued",

            "progress": 0,

            "processed_frames": 0,

            "total_frames": 0,

            "message": "Video uploaded. Starting inference."

        }


    # --------------------------------------------------------
    # START BACKGROUND THREAD
    # --------------------------------------------------------

    thread = threading.Thread(

        target=worker,

        args=(jid, src, out_name, cfg),

        daemon=True

    )

    thread.start()


    return jsonify(

        job_id=jid

    ), 202


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/api/status/<jid>")
def status(jid):

    with lock:

        if jid not in jobs:

            abort(404)


        return jsonify(jobs[jid])


# ============================================================
# OUTPUT FILES
# ============================================================

@app.get("/outputs/<path:name>")
def output(name):

    return send_from_directory(
        OUTPUTS,
        name
    )


# ============================================================
# UPLOAD ERROR
# ============================================================

@app.errorhandler(413)
def too_big(_):

    return jsonify(

        error=(
            "Upload exceeds 1 GB limit. "
            "Increase MAX_UPLOAD_BYTES in app.py."
        )

    ), 413


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print("\n===================================")

    print("       ROADSCENE SERVER")

    print("===================================")

    print("Upload directory:", UPLOADS)

    print("Output directory:", OUTPUTS)

    print("Maximum upload size: 1 GB")

    print("\nOpen http://127.0.0.1:5000")

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )