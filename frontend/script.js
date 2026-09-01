const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictedDigitEl = document.getElementById('predicted-digit');
const statusEl = document.getElementById('prediction-status');
const listEl = document.getElementById('probability-list');

const BRUSH_SIZE = 16;
const LIVE_DEBOUNCE_MS = 350;

let isDrawing = false;
let hasDrawnAnything = false;
let lastX = 0;
let lastY = 0;
let debounceTimer = null;

// ---------- canvas setup ----------

function resetCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = BRUSH_SIZE;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#EDEDE3';
}

resetCanvas();

// ---------- probability row setup ----------

function buildProbabilityRows() {
    listEl.innerHTML = '';
    for (let digit = 0; digit <= 9; digit++) {
        const row = document.createElement('li');
        row.className = 'probability-row';
        row.id = `row-${digit}`;
        row.innerHTML = `
            <span class="digit-label">${digit}</span>
            <span class="bar-track"><span class="bar-fill" id="bar-${digit}"></span></span>
            <span class="pct" id="pct-${digit}">0%</span>
        `;
        listEl.appendChild(row);
    }
}

buildProbabilityRows();

// ---------- pointer position helpers (mouse + touch share this) ----------

function getPointerPos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const point = event.touches ? event.touches[0] : event;

    return {
        x: (point.clientX - rect.left) * scaleX,
        y: (point.clientY - rect.top) * scaleY,
    };
}

function startStroke(event) {
    event.preventDefault();
    isDrawing = true;
    hasDrawnAnything = true;
    const pos = getPointerPos(event);
    lastX = pos.x;
    lastY = pos.y;
}

function drawStroke(event) {
    if (!isDrawing) return;
    event.preventDefault();

    const pos = getPointerPos(event);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastX = pos.x;
    lastY = pos.y;

    scheduleLivePrediction();
}

function endStroke(event) {
    if (!isDrawing) return;
    event.preventDefault();
    isDrawing = false;

    clearTimeout(debounceTimer);
    runPrediction();
}

// mouse
canvas.addEventListener('mousedown', startStroke);
canvas.addEventListener('mousemove', drawStroke);
window.addEventListener('mouseup', endStroke);

// touch
canvas.addEventListener('touchstart', startStroke, { passive: false });
canvas.addEventListener('touchmove', drawStroke, { passive: false });
canvas.addEventListener('touchend', endStroke, { passive: false });
canvas.addEventListener('touchcancel', endStroke, { passive: false });

// ---------- clear ----------

clearBtn.addEventListener('click', () => {
    resetCanvas();
    hasDrawnAnything = false;
    clearTimeout(debounceTimer);
    predictedDigitEl.textContent = '-';
    predictedDigitEl.classList.remove('is-active');
    statusEl.textContent = 'Draw a digit to see a prediction';
    for (let digit = 0; digit <= 9; digit++) {
        document.getElementById(`bar-${digit}`).style.width = '0%';
        document.getElementById(`pct-${digit}`).textContent = '0%';
        document.getElementById(`row-${digit}`).classList.remove('is-top');
    }
});

// ---------- prediction ----------

function scheduleLivePrediction() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPrediction, LIVE_DEBOUNCE_MS);
}

async function runPrediction() {
    if (!hasDrawnAnything) return;

    statusEl.textContent = 'Thinking...';

    try {
        const imageData = canvas.toDataURL('image/png');
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        renderPrediction(data.prediction, data.probabilities);
    } catch (err) {
        statusEl.textContent = "Couldn't reach the network - is the server running?";
        console.error('Prediction request failed:', err);
    }
}

function renderPrediction(predictedDigit, probabilities) {
    predictedDigitEl.textContent = String(predictedDigit);
    predictedDigitEl.classList.remove('is-active');
    void predictedDigitEl.offsetWidth;
    predictedDigitEl.classList.add('is-active');

    const topPct = Math.round(probabilities[predictedDigit] * 100);
    statusEl.textContent = `${topPct}% confident`;

    probabilities.forEach((probability, digit) => {
        const pct = Math.round(probability * 100);
        document.getElementById(`bar-${digit}`).style.width = `${pct}%`;
        document.getElementById(`pct-${digit}`).textContent = `${pct}%`;
        document.getElementById(`row-${digit}`).classList.toggle('is-top', digit === predictedDigit);
    });
}
