
'use strict';

const $ = id => document.getElementById(id);
const labels = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

const video = $('video');
const canvas = $('view');
const ctx = canvas.getContext('2d');

const crop = document.createElement('canvas');
crop.width = crop.height = 28;

const cc = crop.getContext('2d', {
    willReadFrequently: true
});

let model, hands;
let stream = null;
let running = false;
let busy = false;
let queue = [];
let last = '';
let candidate = '';
let generation = 0;

function status(s) {
    $('status').textContent = s;
}

function reset() {
    queue = [];
    last = '';
    candidate = '';

    $('letter').textContent = '—';
    $('confidence').textContent = 'Confidence: —';
    $('bar').value = 0;
    $('add').disabled = true;
    $('stable').textContent =
        'Hold a gesture steadily or upload a photo.';
}

// Load original trained model
async function load() {
    const r = await Promise.all([
        fetch('weights.json'),
        fetch('asl_weights.bin')
    ]);

    if (r.some(x => !x.ok)) {
        throw Error('Model files not found');
    }

    const specs = await r[0].json();
    const bytes = await r[1].arrayBuffer();

    model = tf.sequential();

    const l = tf.layers;

    model.add(l.conv2d({
        inputShape: [28, 28, 1],
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(l.batchNormalization({
        epsilon: 0.001
    }));

    model.add(l.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    model.add(l.dropout({
        rate: 0.3
    }));

    model.add(l.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(l.batchNormalization({
        epsilon: 0.001
    }));

    model.add(l.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    model.add(l.dropout({
        rate: 0.5
    }));

    model.add(l.flatten());

    model.add(l.dense({
        units: 64,
        activation: 'relu'
    }));

    model.add(l.batchNormalization({
        epsilon: 0.001
    }));

    model.add(l.dropout({
        rate: 0.3
    }));

    model.add(l.dense({
        units: 24,
        activation: 'softmax'
    }));

    // Read original model weights
    let offset = 0;

    const weights = specs.map(s => {
        const t = tf.tensor(
            new Float32Array(bytes, offset, s.count),
            s.shape
        );

        offset += s.count * 4;

        return t;
    });

    if (offset !== bytes.byteLength) {
        throw Error('Weight file length mismatch');
    }

    // FIX: Load weights individually for each layer
    // instead of using model.setWeights().

    let index = 0;

    try {
        for (const layer of model.layers) {
            const count = layer.weights.length;

            if (!count) continue;

            const group = weights.slice(
                index,
                index + count
            );

            if (
                group.length !== count ||
                group.some((t, i) =>
                    JSON.stringify(t.shape) !==
                    JSON.stringify(layer.weights[i].shape)
                )
            ) {
                throw Error(
                    'Weight shape mismatch in ' + layer.name
                );
            }

            layer.setWeights(group);

            index += count;
        }

        if (index !== weights.length) {
            throw Error('Unused model weights');
        }

    } finally {
        weights.forEach(t => t.dispose());
    }

    // Validate model output
    tf.tidy(() => {
        const p = model.predict(
            tf.zeros([1, 28, 28, 1])
        ).dataSync();

        if (
            p.length !== 24 ||
            !p.every(Number.isFinite)
        ) {
            throw Error('Model validation failed');
        }
    });

    // Initialise MediaPipe hand tracking
    hands = new Hands({
        locateFile: f =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });

    hands.onResults(handle);

    await hands.initialize();

    $('start').disabled = false;
    $('upload').disabled = false;

    status(
        'Ready. Start the camera or upload a hand image.'
    );
}

// Process hand detection results
function handle(results) {
    const im = results.image;

    canvas.width =
        im.videoWidth || im.naturalWidth || im.width;

    canvas.height =
        im.videoHeight || im.naturalHeight || im.height;

    ctx.drawImage(
        im,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const lm = results.multiHandLandmarks?.[0];

    if (!lm) {
        reset();
        return;
    }

    const w = canvas.width;
    const h = canvas.height;

    const x = Math.max(
        0,
        Math.floor(Math.min(...lm.map(p => p.x)) * w)
    );

    const y = Math.max(
        0,
        Math.floor(Math.min(...lm.map(p => p.y)) * h)
    );

    const x2 = Math.min(
        w,
        Math.floor(Math.max(...lm.map(p => p.x)) * w)
    );

    const y2 = Math.min(
        h,
        Math.floor(Math.max(...lm.map(p => p.y)) * h)
    );

    if (x2 <= x || y2 <= y) {
        reset();
        return;
    }

    // Crop and resize hand image
    cc.drawImage(
        canvas,
        x,
        y,
        x2 - x,
        y2 - y,
        0,
        0,
        28,
        28
    );

    const rgba = cc.getImageData(0, 0, 28, 28).data;

    // Convert to normalised grayscale
    const gray = Float32Array.from(
        { length: 784 },
        (_, i) => (
            0.299 * rgba[i * 4] +
            0.587 * rgba[i * 4 + 1] +
            0.114 * rgba[i * 4 + 2]
        ) / 255
    );

    // Run prediction
    const p = tf.tidy(() =>
        Array.from(
            model.predict(
                tf.tensor4d(gray, [1, 28, 28, 1])
            ).dataSync()
        )
    );

    const conf = Math.max(...p);

    const letter = conf > 0.8
        ? labels[p.indexOf(conf)]
        : 'Unknown';

    candidate = letter === 'Unknown' ? '' : letter;

    $('letter').textContent = letter;

    $('confidence').textContent =
        `Confidence: ${(conf * 100).toFixed(1)}%`;

    $('bar').value = conf;
    $('add').disabled = !candidate;

    // Draw hand bounding box
    ctx.strokeStyle = '#35e5b1';
    ctx.lineWidth = 3;

    ctx.strokeRect(
        x,
        y,
        x2 - x,
        y2 - y
    );

    // Draw hand landmarks
    for (const point of lm) {
        ctx.beginPath();

        ctx.arc(
            point.x * w,
            point.y * h,
            3,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = '#35e5b1';
        ctx.fill();
    }

    // Stabilise predictions
    if (running) {
        queue.push(letter);

        if (queue.length > 15) {
            queue.shift();
        }

        const stable = labels.find(
            a => queue.filter(v => v === a).length > 10
        );

        if (stable && stable !== last) {
            $('buffer').value += stable;
            last = stable;
        }

        $('stable').textContent = stable
            ? `Stable gesture: ${stable}`
            : 'Hold gesture steadily for automatic entry.';

    } else {
        $('stable').textContent = candidate
            ? 'Image prediction ready. Select Add letter.'
            : 'Try another image.';
    }
}

// Webcam recognition loop
async function loop(g) {
    if (!running || g !== generation) return;

    if (!busy && video.readyState >= 2) {
        busy = true;

        try {
            await hands.send({
                image: video
            });

        } catch (e) {
            stop();
            status('Recognition error: ' + e.message);

        } finally {
            busy = false;
        }
    }

    if (running && g === generation) {
        requestAnimationFrame(() => loop(g));
    }
}

// Stop webcam
function stop() {
    generation++;
    running = false;

    stream?.getTracks().forEach(t => t.stop());

    stream = null;
    video.srcObject = null;

    $('start').disabled = false;
    $('stop').disabled = true;
    $('upload').disabled = false;

    reset();
}

// Start webcam
$('start').onclick = async () => {
    if (busy) return;

    $('start').disabled = true;
    $('upload').disabled = true;

    const g = ++generation;

    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw Error(
                'Camera requires HTTPS or localhost'
            );
        }

        const s =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user'
                },
                audio: false
            });

        if (g !== generation) {
            s.getTracks().forEach(t => t.stop());
            return;
        }

        stream = s;
        video.srcObject = s;

        await video.play();

        running = true;

        $('stop').disabled = false;

        status(
            'Camera active. Show one hand with good lighting.'
        );

        loop(g);

    } catch (e) {
        stop();

        status(
            'Camera unavailable: ' +
            e.message +
            '. Try uploading an image.'
        );
    }
};

// Stop button
$('stop').onclick = () => {
    stop();
    status('Camera stopped.');
};

// Image upload
$('upload').onchange = async e => {
    const file = e.target.files[0];

    if (!file || busy) return;

    stop();

    busy = true;

    $('start').disabled = true;
    $('upload').disabled = true;

    const url = URL.createObjectURL(file);

    try {
        const im = new Image();

        im.src = url;

        await im.decode();

        await hands.send({
            image: im
        });

        status('Image processed.');

    } catch (err) {
        status('Image error: ' + err.message);

    } finally {
        URL.revokeObjectURL(url);

        busy = false;

        $('start').disabled = false;
        $('upload').disabled = false;

        e.target.value = '';
    }
};

// Add detected letter
$('add').onclick = () => {
    if (candidate) {
        $('buffer').value += candidate;
    }
};

// Add space
$('space').onclick = () => {
    $('buffer').value += ' ';
};

// Backspace
$('undo').onclick = () => {
    $('buffer').value =
        $('buffer').value.slice(0, -1);
};

// Clear text
$('clear').onclick = () => {
    $('buffer').value = '';
    last = '';
    queue = [];
};

// Text-to-speech
$('speak').onclick = () => {
    if (!('speechSynthesis' in window)) {
        status(
            'Speech is not supported by this browser.'
        );

        return;
    }

    if ($('buffer').value.trim()) {
        speechSynthesis.cancel();

        const u = new SpeechSynthesisUtterance(
            $('buffer').value
        );

        u.lang = 'en-US';

        speechSynthesis.speak(u);
    }
};

// Release camera when leaving the page
window.addEventListener('pagehide', () =>
    stream?.getTracks().forEach(t => t.stop())
);

// Initialise application
load().catch(e =>
    status(
        'Unable to initialise demo: ' +
        e.message +
        '. Check internet access and model files.'
    )
);
