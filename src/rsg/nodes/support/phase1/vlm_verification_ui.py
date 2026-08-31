"""Web UI for manual verification of VLM predictions."""

import json
from pathlib import Path
from typing import Dict, Any, List
import csv


class VerificationUIGenerator:
    """Generate interactive web UI for VLM prediction verification."""

    def __init__(self, session_dir: Path):
        """Initialize UI generator.

        Args:
            session_dir: Session directory containing crops and outputs
        """
        self.session_dir = Path(session_dir)
        self.crops_dir = self.session_dir / "crops"
        self.vlm_outputs_file = self.session_dir / "vlm_outputs.jsonl"
        self.verification_file = self.session_dir / "verification_results.csv"

    def generate_html(self, max_samples: int = 50) -> str:
        """Generate interactive HTML verification interface.

        Args:
            max_samples: Maximum number of samples to display

        Returns:
            HTML string
        """
        # Load VLM outputs
        outputs = self._load_vlm_outputs()
        samples = outputs[:max_samples]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Prediction Verification</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{ font-size: 28px; margin-bottom: 10px; color: #333; }}
        .header-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat {{ background: #f8f9fa; padding: 10px; border-radius: 4px; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 5px; }}

        .samples-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}

        .sample-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .sample-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}

        .sample-card.verified {{ border-left: 4px solid #27ae60; }}
        .sample-card.unverified {{ border-left: 4px solid #95a5a6; }}

        .crop-image {{
            width: 100%;
            height: 300px;
            object-fit: cover;
            background: #f0f0f0;
        }}

        .sample-info {{
            padding: 15px;
        }}

        .sample-id {{
            font-size: 12px;
            color: #7f8c8d;
            font-family: monospace;
            margin-bottom: 10px;
        }}

        .vlm-prediction {{
            background: #ecf0f1;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
        }}

        .pred-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 4px; }}
        .pred-class {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .pred-confidence {{
            font-size: 14px;
            color: #3498db;
            margin-top: 4px;
        }}
        .pred-time {{
            font-size: 12px;
            color: #95a5a6;
            margin-top: 4px;
        }}

        .verification-form {{
            display: grid;
            gap: 10px;
        }}

        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}

        label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            font-weight: 600;
        }}

        input[type="text"], select, textarea {{
            padding: 8px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
        }}

        textarea {{
            resize: vertical;
            min-height: 50px;
        }}

        .buttons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}

        button {{
            padding: 10px;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .btn-correct {{
            background: #27ae60;
            color: white;
        }}
        .btn-correct:hover {{ background: #229954; }}

        .btn-incorrect {{
            background: #e74c3c;
            color: white;
        }}
        .btn-incorrect:hover {{ background: #c0392b; }}

        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-top: 10px;
        }}

        .status-verified {{ background: #d5f4e6; color: #27ae60; }}
        .status-unverified {{ background: #ecf0f1; color: #95a5a6; }}

        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2980b9);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 VLM Prediction Verification</h1>
            <div class="header-stats">
                <div class="stat">
                    <div class="stat-label">Total Samples</div>
                    <div class="stat-value">{len(samples)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Verified</div>
                    <div class="stat-value verified-count">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Accuracy</div>
                    <div class="stat-value accuracy-count">-</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Avg Process Time</div>
                    <div class="stat-value avg-time">-</div>
                </div>
            </div>
        </header>

        <div class="samples-container">
"""

        for i, output in enumerate(samples, 1):
            crop_file = output.get("crop_filename", "unknown.jpg")
            crop_path = f"crops/{crop_file}"
            vlm_class = output.get("vlm_class", "unknown")
            confidence = output.get("vlm_confidence", 0.0)
            proc_time = output.get("vlm_processing_time_ms", 0.0)
            object_id = output.get("object_id", "N/A")

            html += f"""            <div class="sample-card unverified" id="card-{i}">
                <img src="{crop_path}" alt="Crop {i}" class="crop-image" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22><rect fill=%22%23f0f0f0%22 width=%22400%22 height=%22300%22/><text x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22>Image not found</text></svg>'">

                <div class="sample-info">
                    <div class="sample-id">Object #{object_id}</div>

                    <div class="vlm-prediction">
                        <div class="pred-label">VLM Prediction</div>
                        <div class="pred-class">{vlm_class}</div>
                        <div class="pred-confidence">Confidence: {confidence:.1%}</div>
                        <div class="pred-time">Process time: {proc_time:.2f} ms</div>
                    </div>

                    <div class="verification-form">
                        <div class="form-group">
                            <label>Actual Class</label>
                            <input type="text" class="actual-class" placeholder="Enter correct class name" data-sample="{i}">
                        </div>

                        <div class="form-group">
                            <label>Your Confidence</label>
                            <select class="your-confidence" data-sample="{i}">
                                <option value="">Select confidence</option>
                                <option value="0.1">Very Uncertain (10%)</option>
                                <option value="0.3">Uncertain (30%)</option>
                                <option value="0.5">Neutral (50%)</option>
                                <option value="0.7">Confident (70%)</option>
                                <option value="0.9">Very Confident (90%)</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label>Notes</label>
                            <textarea class="notes" placeholder="Any observations..." data-sample="{i}"></textarea>
                        </div>

                        <div class="buttons">
                            <button class="btn-correct" onclick="recordResult({i}, true)">✓ Correct</button>
                            <button class="btn-incorrect" onclick="recordResult({i}, false)">✗ Incorrect</button>
                        </div>

                        <div id="status-{i}" class="status-badge status-unverified">Not verified</div>
                    </div>
                </div>
            </div>
"""

        html += """        </div>
    </div>

    <script>
        const results = {{}};

        function recordResult(sampleId, isCorrect) {{
            const actualClass = document.querySelector(`.actual-class[data-sample="{sampleId}"]`).value;
            const confidence = document.querySelector(`.your-confidence[data-sample="{sampleId}"]`).value;
            const notes = document.querySelector(`.notes[data-sample="{sampleId}"]`).value;

            if (!actualClass || !confidence) {{
                alert('Please fill in all fields');
                return;
            }}

            results[sampleId] = {{
                isCorrect: isCorrect,
                actualClass: actualClass,
                confidence: confidence,
                notes: notes
            }};

            // Update UI
            const card = document.getElementById(`card-{sampleId}`);
            card.classList.remove('unverified');
            card.classList.add('verified');

            const status = document.getElementById(`status-{sampleId}`);
            status.className = 'status-badge status-verified';
            status.textContent = isCorrect ? '✓ Correct' : '✗ Incorrect';

            // Update stats
            updateStats();

            // Download updated CSV
            downloadResults();
        }}

        function updateStats() {{
            const verified = Object.keys(results).length;
            const total = {len(samples)};
            const correct = Object.values(results).filter(r => r.isCorrect).length;
            const accuracy = verified > 0 ? (correct / verified * 100).toFixed(1) : '-';

            document.querySelector('.verified-count').textContent = verified;
            document.querySelector('.accuracy-count').textContent = accuracy + '%';

            const avgTime = {sum(o.get('vlm_processing_time_ms', 0) for o in samples) / len(samples) if samples else 0:.2f};
            document.querySelector('.avg-time').textContent = avgTime.toFixed(2) + ' ms';
        }}

        function downloadResults() {{
            // Auto-download as user verifies
            const csv = generateCSV();
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'verification_results_LIVE.csv';
            // Uncomment to auto-download:
            // a.click();
        }}

        function generateCSV() {{
            let csv = 'sample_id,is_correct,actual_class,confidence,notes\\n';
            for (const [id, result] of Object.entries(results)) {{
                csv += `${{id}},${{result.isCorrect}},${{result.actualClass}},${{result.confidence}},\\"${{result.notes}}\\"\\n`;
            }}
            return csv;
        }}

        // Save results periodically
        setInterval(() => {{
            if (Object.keys(results).length > 0) {{
                localStorage.setItem('vlmVerificationResults', JSON.stringify(results));
            }}
        }}, 5000);
    </script>
</body>
</html>
"""
        return html

    def _load_vlm_outputs(self) -> List[Dict[str, Any]]:
        """Load VLM outputs from JSONL file."""
        outputs = []
        try:
            with open(self.vlm_outputs_file, "r") as f:
                for line in f:
                    try:
                        outputs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        return outputs

    def save_html(self, filename: str = "verification_ui.html") -> Path:
        """Save HTML UI to file.

        Args:
            filename: Output filename

        Returns:
            Path to saved HTML file
        """
        output_file = self.session_dir / filename
        html = self.generate_html()
        with open(output_file, "w") as f:
            f.write(html)
        return output_file
