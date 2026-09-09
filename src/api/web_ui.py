"""
Presentation-ready Web UI for Audio Anti-Spoofing Detection.
Served at http://localhost:8000/
"""

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoiceGuard AI — Deepfake & Voice Clone Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.12);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-glow: rgba(59, 130, 246, 0.15);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
    }

    .container {
      width: 100%;
      max-width: 820px;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 36px;
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      color: #60a5fa;
      margin-bottom: 16px;
    }

    .badge-pill span.dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #3b82f6;
      box-shadow: 0 0 10px #3b82f6;
    }

    h1 {
      font-size: 36px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 10px;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    p.subtitle {
      color: var(--text-muted);
      font-size: 16px;
      max-width: 580px;
      margin: 0 auto;
      line-height: 1.5;
    }

    /* Upload Box */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 32px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
      position: relative;
      overflow: hidden;
    }

    .dropzone {
      border: 2px dashed #2d3748;
      border-radius: 16px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(17, 24, 39, 0.6);
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(59, 130, 246, 0.04);
      transform: translateY(-2px);
    }

    .upload-icon {
      width: 54px;
      height: 54px;
      margin: 0 auto 16px;
      background: rgba(59, 130, 246, 0.1);
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--primary);
    }

    .upload-icon svg {
      width: 28px;
      height: 28px;
    }

    .dropzone h3 {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 6px;
    }

    .dropzone p {
      color: var(--text-muted);
      font-size: 13px;
    }

    input[type="file"] {
      display: none;
    }

    /* File Preview */
    .file-preview {
      display: none;
      margin-top: 20px;
      padding: 16px 20px;
      background: #1a2234;
      border: 1px solid #28354d;
      border-radius: 12px;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .file-preview.active {
      display: flex;
    }

    .file-info {
      display: flex;
      align-items: center;
      gap: 12px;
      overflow: hidden;
    }

    .file-icon {
      width: 40px;
      height: 40px;
      background: #232f48;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #60a5fa;
      flex-shrink: 0;
    }

    .file-details {
      overflow: hidden;
    }

    .file-name {
      font-size: 14px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 280px;
    }

    .file-size {
      font-size: 12px;
      color: var(--text-muted);
    }

    audio {
      height: 36px;
      max-width: 240px;
      border-radius: 8px;
    }

    /* Analyze Button */
    .btn-analyze {
      width: 100%;
      margin-top: 24px;
      padding: 16px;
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
    }

    .btn-analyze:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 12px 24px -5px rgba(59, 130, 246, 0.6);
    }

    .btn-analyze:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    /* Results Card */
    .result-section {
      display: none;
      margin-top: 32px;
      animation: fadeIn 0.4s ease forwards;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-card {
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      border: 1px solid;
    }

    .result-card.real {
      background: var(--success-bg);
      border-color: rgba(16, 185, 129, 0.4);
    }

    .result-card.fake {
      background: var(--danger-bg);
      border-color: rgba(239, 68, 68, 0.4);
    }

    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }

    .verdict-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.01em;
    }

    .verdict-badge.real {
      color: var(--success);
    }

    .verdict-badge.fake {
      color: var(--danger);
    }

    .verdict-icon {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
    }

    .verdict-icon.real {
      background: var(--success);
    }

    .verdict-icon.fake {
      background: var(--danger);
    }

    .confidence-meter {
      margin-top: 12px;
    }

    .meter-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 8px;
    }

    .meter-bar {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 9999px;
      overflow: hidden;
    }

    .meter-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .meter-fill.real {
      background: var(--success);
    }

    .meter-fill.fake {
      background: var(--danger);
    }

    /* Diagnostics */
    .diagnostics-box {
      background: #141d2f;
      border: 1px solid #1f2a40;
      border-radius: 12px;
      padding: 18px 20px;
    }

    .diagnostics-box h4 {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 12px;
    }

    .flag-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .flag-pill {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: #cbd5e1;
      background: #19243a;
      padding: 8px 12px;
      border-radius: 8px;
      border-left: 3px solid #64748b;
    }

    .flag-pill.suspicious {
      border-left-color: var(--danger);
      background: rgba(239, 68, 68, 0.08);
    }

    .flag-pill.normal {
      border-left-color: var(--success);
      background: rgba(16, 185, 129, 0.08);
    }

    /* Loading Spinner */
    .spinner {
      display: none;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Footer */
    footer {
      margin-top: 36px;
      text-align: center;
      color: #64748b;
      font-size: 13px;
    }
    
    footer a {
      color: #94a3b8;
      text-decoration: none;
      transition: color 0.2s;
    }
    
    footer a:hover {
      color: #3b82f6;
    }
  </style>
</head>
<body>

  <div class="container">
    <header>
      <div class="badge-pill">
        <span class="dot"></span> Ensemble CNN + Bi-LSTM Neural Detector
      </div>
      <h1>Audio Anti-Spoofing & Deepfake Detector</h1>
      <p class="subtitle">Upload any audio file to analyze acoustic forensics, neural vocoder phase artifacts, and verify human authenticity.</p>
    </header>

    <div class="card">
      <div class="dropzone" id="dropzone">
        <div class="upload-icon">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
        </div>
        <h3>Drop your audio file here, or browse</h3>
        <p>Supports MP3, WAV, FLAC, OGG, M4A (Max 10MB)</p>
        <input type="file" id="fileInput" accept=".wav,.mp3,.flac,.ogg,.m4a">
      </div>

      <div class="file-preview" id="filePreview">
        <div class="file-info">
          <div class="file-icon">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>
            </svg>
          </div>
          <div class="file-details">
            <div class="file-name" id="fileName">sample.wav</div>
            <div class="file-size" id="fileSize">0 KB</div>
          </div>
        </div>
        <audio id="audioPlayer" controls></audio>
      </div>

      <button class="btn-analyze" id="analyzeBtn" disabled>
        <div class="spinner" id="spinner"></div>
        <span id="btnText">Analyze Audio Forensics</span>
      </button>

      <!-- Results Section -->
      <div class="result-section" id="resultSection">
        <div class="result-card" id="resultCard">
          <div class="result-header">
            <div class="verdict-badge" id="verdictBadge">
              <div class="verdict-icon" id="verdictIcon">✓</div>
              <span id="verdictText">GENUINE HUMAN SPEECH</span>
            </div>
            <span style="font-size: 13px; color: var(--text-muted);" id="latencyText">Latency: 28ms</span>
          </div>

          <div class="confidence-meter">
            <div class="meter-label">
              <span>Model Confidence</span>
              <span id="confidenceText">99.2%</span>
            </div>
            <div class="meter-bar">
              <div class="meter-fill" id="meterFill" style="width: 0%;"></div>
            </div>
          </div>
        </div>

        <div class="diagnostics-box">
          <h4>Forensic Diagnostics & Artifacts</h4>
          <div class="flag-list" id="flagList">
            <!-- Populated dynamically -->
          </div>
        </div>
      </div>
    </div>

    <footer>
      VoiceGuard Deepfake Defense &bull; <a href="/docs" target="_blank">View Technical API Docs (Swagger)</a>
    </footer>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const audioPlayer = document.getElementById('audioPlayer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const verdictBadge = document.getElementById('verdictBadge');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictText = document.getElementById('verdictText');
    const latencyText = document.getElementById('latencyText');
    const confidenceText = document.getElementById('confidenceText');
    const meterFill = document.getElementById('meterFill');
    const flagList = document.getElementById('flagList');

    let selectedFile = null;

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });

    function handleFile(file) {
      selectedFile = file;
      fileName.textContent = file.name;
      fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
      
      const audioUrl = URL.createObjectURL(file);
      audioPlayer.src = audioUrl;

      filePreview.classList.add('active');
      analyzeBtn.disabled = false;
      resultSection.style.display = 'none';
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      analyzeBtn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing Spectrogram & Phase...';

      const formData = new FormData();
      formData.append('audio', selectedFile);

      try {
        const response = await fetch('/api/v1/detect', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data);
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        analyzeBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Analyze Audio Forensics';
      }
    });

    function displayResults(data) {
      resultSection.style.display = 'block';

      const isReal = data.classification === 'REAL';
      const confPercent = (data.confidence * 100).toFixed(1) + '%';

      // Reset classes
      resultCard.className = 'result-card ' + (isReal ? 'real' : 'fake');
      verdictBadge.className = 'verdict-badge ' + (isReal ? 'real' : 'fake');
      verdictIcon.className = 'verdict-icon ' + (isReal ? 'real' : 'fake');
      meterFill.className = 'meter-fill ' + (isReal ? 'real' : 'fake');

      // Update text
      verdictIcon.textContent = isReal ? '✓' : '⚠';
      verdictText.textContent = isReal ? 'VERIFIED REAL HUMAN SPEECH' : 'AI VOICE CLONE / DEEPFAKE DETECTED';
      latencyText.textContent = `Latency: ${data.processing_time_ms.toFixed(1)}ms`;
      confidenceText.textContent = confPercent;

      // Animate progress bar
      setTimeout(() => {
        meterFill.style.width = confPercent;
      }, 50);

      // Populate flags
      flagList.innerHTML = '';
      if (data.diagnostic_flags && data.diagnostic_flags.length > 0) {
        data.diagnostic_flags.forEach(flag => {
          const div = document.createElement('div');
          const isSuspicious = !isReal || flag.toLowerCase().includes('abnormal') || flag.toLowerCase().includes('detected');
          div.className = 'flag-pill ' + (isSuspicious ? 'suspicious' : 'normal');
          div.innerHTML = `<span>${isSuspicious ? '🔍' : '✓'}</span> <span>${flag}</span>`;
          flagList.appendChild(div);
        });
      } else {
        const div = document.createElement('div');
        div.className = 'flag-pill normal';
        div.innerHTML = `<span>✓</span> <span>Acoustic and phase patterns are consistent with authentic human speech.</span>`;
        flagList.appendChild(div);
      }
    }
  </script>
</body>
</html>
"""
