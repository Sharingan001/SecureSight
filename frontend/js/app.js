/**
 * SecureSight — Dashboard Application
 * Professional forensic interface with auth, smooth animations.
 */

const API = window.location.origin + '/api/v1';
let current = null;

/* ═══ AUTH MANAGER ═════════════════════════════════════════ */
const Auth = {
    getToken() {
        return localStorage.getItem('ss_token');
    },
    setToken(token) {
        localStorage.setItem('ss_token', token);
    },
    clearToken() {
        localStorage.removeItem('ss_token');
        localStorage.removeItem('ss_user');
    },
    getUser() {
        try { return JSON.parse(localStorage.getItem('ss_user')); }
        catch { return null; }
    },
    setUser(user) {
        localStorage.setItem('ss_user', JSON.stringify(user));
    },
    isLoggedIn() {
        return !!this.getToken();
    },
    headers() {
        const h = { 'Accept': 'application/json' };
        const token = this.getToken();
        if (token) h['Authorization'] = `Bearer ${token}`;
        return h;
    },
    async logout() {
        if (this.getToken()) {
            try {
                await fetch(`${API}/auth/logout`, {
                    method: 'POST',
                    headers: this.headers()
                });
            } catch (e) {
                console.warn("Logout API failed, forcing local logout", e);
            }
        }
        this.clearToken();
        showAuthModal();
        updateAuthUI();
    }
};

/** Authenticated fetch — adds Bearer token, handles 401 */
async function authFetch(url, opts = {}) {
    opts.headers = { ...Auth.headers(), ...(opts.headers || {}) };
    const res = await fetch(url, opts);
    if (res.status === 401) {
        Auth.clearToken();
        showAuthModal();
        throw new Error('Session expired — please log in again');
    }
    return res;
}

/* ═══ AUTH UI ══════════════════════════════════════════════ */
function showAuthModal() {
    document.getElementById('authOverlay').classList.add('active');
    document.getElementById('authError').textContent = '';
}

function hideAuthModal() {
    document.getElementById('authOverlay').classList.remove('active');
}

function updateAuthUI() {
    const userInfo = document.getElementById('userInfo');
    const loginBtn = document.getElementById('loginBtn');

    if (Auth.isLoggedIn()) {
        const user = Auth.getUser();
        userInfo.textContent = user?.email?.split('@')[0] || 'User';
        userInfo.style.display = 'inline';
        loginBtn.textContent = '↗ Logout';
        loginBtn.onclick = () => Auth.logout();
    } else {
        userInfo.style.display = 'none';
        loginBtn.textContent = '→ Login';
        loginBtn.onclick = () => showAuthModal();
    }
}

// Auth form tabs
document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const isLogin = tab.dataset.mode === 'login';
        document.getElementById('authTitle').textContent = isLogin ? 'Welcome Back' : 'Create Account';
        document.getElementById('nameField').style.display = isLogin ? 'none' : 'block';
        document.getElementById('authSubmit').textContent = isLogin ? 'Sign In' : 'Create Account';
        document.getElementById('authForm').dataset.mode = tab.dataset.mode;
        document.getElementById('authError').textContent = '';
    });
});

// Auth form submit
document.getElementById('authForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const mode = form.dataset.mode || 'login';
    const email = document.getElementById('authEmail').value.trim();
    const password = document.getElementById('authPassword').value;
    const errEl = document.getElementById('authError');
    const submitBtn = document.getElementById('authSubmit');

    errEl.textContent = '';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Please wait...';

    try {
        const endpoint = mode === 'login' ? '/auth/login' : '/auth/register';
        const body = { email, password };
        if (mode === 'register') {
            body.full_name = document.getElementById('authName').value.trim();
        }

        const res = await fetch(`${API}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Authentication failed');
        }

        if (data.requires_2fa) {
            errEl.textContent = '2FA required — feature coming to UI soon. Contact admin.';
            errEl.style.color = 'var(--amber)';
            return;
        }

        if (!data.access_token) {
            throw new Error('No access token received');
        }

        Auth.setToken(data.access_token);
        Auth.setUser({ email: data.email, role: data.role, user_id: data.user_id });

        hideAuthModal();
        updateAuthUI();
        loadHistory();

    } catch (err) {
        errEl.textContent = err.message;
        errEl.style.color = '';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = mode === 'login' ? 'Sign In' : 'Create Account';
    }
});

/* ═══ NAVIGATION ═══════════════════════════════════════════ */
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
});

function switchView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));

    const view = document.getElementById(`v-${name}`);
    const btn = document.querySelector(`.nav-item[data-view="${name}"]`);
    if (view) view.classList.add('active');
    if (btn) btn.classList.add('active');
}

/* Header scroll effect */
window.addEventListener('scroll', () => {
    const header = document.getElementById('appHeader');
    header.classList.toggle('scrolled', window.scrollY > 20);
});

/* ═══ UPLOAD ═══════════════════════════════════════════════ */
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const progressPct = document.getElementById('progressPct');

uploadBox.addEventListener('click', () => {
    if (!Auth.isLoggedIn()) { showAuthModal(); return; }
    fileInput.click();
});
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    if (!Auth.isLoggedIn()) { showAuthModal(); return; }
    if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) processFile(fileInput.files[0]); });

async function processFile(file) {
    if (!Auth.isLoggedIn()) { showAuthModal(); return; }
    if (file.size > 500 * 1024 * 1024) { alert('File exceeds 500 MB limit'); return; }

    // Client-side type validation
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff',
                        'video/mp4', 'video/avi', 'video/mov', 'video/webm', 'video/mkv'];
    if (file.type && !validTypes.includes(file.type)) {
        alert(`Unsupported file type: ${file.type}\nAccepted: JPEG, PNG, WebP, BMP, TIFF, MP4, AVI, MOV, MKV`);
        return;
    }

    uploadBox.style.display = 'none';
    progressWrap.classList.add('active');
    setProgress(3, 'Preparing upload...');

    const fd = new FormData();
    fd.append('file', file);

    try {
        runProgressSim();

        const res = await authFetch(`${API}/analyze`, { method: 'POST', body: fd });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${res.status}`);
        }
        const submitted = await res.json();
        const analysisId = submitted.analysis_id;

        // Poll /progress for real backend stage info (lightweight, no full payload)
        let result = submitted;
        if (result.cached) {
            // Dedup hit — already complete, skip polling
            setProgress(100, 'Cached result returned instantly');
        } else {
            const maxWait = 300;
            let waited = 0;
            while (result.status === 'processing' || result.status === 'pending') {
                if (waited >= maxWait) throw new Error('Analysis timed out after 5 minutes');
                await new Promise(r => setTimeout(r, 2000));
                waited += 2;

                // Lightweight progress check
                try {
                    const progRes = await authFetch(`${API}/results/${analysisId}/progress`);
                    if (progRes.ok) {
                        const prog = await progRes.json();
                        const stageLabels = {
                            queued:        'Waiting in queue...',
                            preprocessing: 'Extracting frames & detecting faces...',
                            pipelines:     'Running AI ensemble pipelines...',
                            visuals:       'Generating heatmaps & GradCAM...',
                            report:        'Building forensic PDF report...',
                            complete:      'Finalising results...',
                        };
                        const label = stageLabels[prog.stage] || prog.detail || 'Analysing...';
                        setProgress(prog.pct ?? 30, label);
                        if (prog.stage) activateStep(
                            { preprocessing:'preprocess', pipelines:'detect',
                              visuals:'forensic', report:'report' }[prog.stage] || prog.stage
                        );
                        if (prog.status === 'completed' || prog.status === 'failed') {
                            result = { status: prog.status };
                            break;
                        }
                    }
                } catch (_) { /* swallow — fall through to next poll */ }
            }
        }

        // Fetch full result once done
        if (!result.cached) {
            const finalRes = await authFetch(`${API}/results/${analysisId}`);
            if (!finalRes.ok) throw new Error('Failed to fetch analysis result');
            result = await finalRes.json();
        } else {
            result = submitted;  // already has full payload from dedup path
        }

        if (result.status === 'failed') {
            throw new Error('Analysis pipeline failed — check server logs');
        }

        current = result;
        setProgress(100, 'Analysis complete');

        setTimeout(() => {
            progressWrap.classList.remove('active');
            uploadBox.style.display = '';
            renderResults(current);
            switchView('results');
            loadHistory();
        }, 600);
    } catch (err) {
        setProgress(0, `Error: ${err.message}`);
        setTimeout(() => { progressWrap.classList.remove('active'); uploadBox.style.display = ''; }, 4000);
    }
}

let simTimer;
function runProgressSim() {
    clearInterval(simTimer);
    let pct = 3;
    const stages = [
        [8, 'preprocess', 'Preprocessing media...'],
        [22, 'detect', 'Running AI detection models...'],
        [35, 'detect', 'EfficientNet-B4 analysis...'],
        [48, 'detect', 'AI Ensemble analysis...'],
        [55, 'forensic', 'Error Level Analysis...'],
        [62, 'forensic', 'Copy-move detection...'],
        [68, 'forensic', 'JPEG ghost detection...'],
        [73, 'forensic', 'Eye reflection check...'],
        [78, 'forensic', 'Shadow & noise analysis...'],
        [84, 'forensic', 'Frequency domain analysis...'],
        [88, 'report', 'Computing ensemble verdict...'],
        [93, 'report', 'Generating heatmaps...'],
        [96, 'report', 'Writing PDF report...'],
    ];
    let si = 0;

    simTimer = setInterval(() => {
        if (pct >= 96) { clearInterval(simTimer); return; }
        pct += Math.random() * 2.5;
        pct = Math.min(pct, 96);

        while (si < stages.length && pct >= stages[si][0]) {
            activateStep(stages[si][1]);
            setProgress(pct, stages[si][2]);
            si++;
        }
        setProgress(pct);
    }, 350);
}

function setProgress(pct, label) {
    progressFill.style.width = `${pct}%`;
    progressPct.textContent = `${Math.round(pct)}%`;
    if (label) progressLabel.textContent = label;
}

function activateStep(name) {
    document.querySelectorAll('.progress-step').forEach(s => {
        if (s.dataset.step === name) s.classList.add('active');
        else if (s.classList.contains('active')) { s.classList.remove('active'); s.classList.add('done'); }
    });
}

/* ═══ RESULTS ══════════════════════════════════════════════ */
function renderResults(data) {
    if (!data) return;

    const score = data.overall_score ?? 0;
    const color = verdictColor(data.verdict);

    // Gauge
    const arc = document.getElementById('gaugeArc');
    const num = document.getElementById('gaugeNum');
    const circ = 2 * Math.PI * 90;
    arc.style.stroke = color;
    arc.style.color = color;
    setTimeout(() => { arc.style.strokeDashoffset = circ * (1 - score / 100); }, 80);
    animateNum(num, 0, score, 1800);

    // Verdict pill
    const pill = document.getElementById('verdictPill');
    pill.textContent = (data.verdict || '—').replace(/_/g, ' ');
    pill.className = `verdict-pill verdict-${data.verdict}`;

    // Evidence info — use textContent to prevent XSS
    document.getElementById('eid').textContent = data.evidence_id || '—';
    document.getElementById('ehash').textContent = data.sha256 || '—';
    document.getElementById('efname').textContent = data.filename || '—';

    // Cached badge — shown when dedup hit (same file already analysed)
    const cachedBadge = document.getElementById('cachedBadge');
    if (cachedBadge) {
        cachedBadge.style.display = data.cached ? 'inline-flex' : 'none';
    }

    // Pipeline list
    const list = document.getElementById('pipelineList');
    list.innerHTML = '';

    const icons = {
        efficientnet: '🧠', xception: '🔍', audio: '🎵', lipsync: '👄',
        ela: '📊', copy_move: '📋', jpeg_ghost: '👻', exif: '📷',
        eye_reflection: '👁️', shadow: '💡', noise: '📡',
        frequency: '📈', biometric: '🧬', temporal: '🎬', ai_ensemble: '🤖',
    };
    const names = {
        efficientnet: 'EfficientNet-B4', xception: 'XceptionNet',
        audio: 'Audio Deepfake', lipsync: 'Lip-Sync Verify',
        ela: 'Error Level Analysis', copy_move: 'Copy-Move Detection',
        jpeg_ghost: 'JPEG Ghost', exif: 'EXIF Metadata',
        eye_reflection: 'Eye Reflection', shadow: 'Shadow / Lighting',
        noise: 'Noise Pattern', frequency: 'Frequency Domain',
        biometric: 'Biometric Mesh', temporal: 'Temporal Consistency',
        ai_ensemble: 'AI Ensemble (3-model)',
    };

    (data.pipeline_scores || []).forEach((p, i) => {
        const pct = (p.score * 100).toFixed(1);
        const barCol = scoreGradient(p.score);

        const el = document.createElement('div');
        el.className = 'pipe-item';

        // Build pipeline item with safe text (no innerHTML for user data)
        const iconEl = document.createElement('div');
        iconEl.className = `pipe-icon t${p.tier}`;
        iconEl.textContent = icons[p.pipeline] || '🔧';

        const infoEl = document.createElement('div');
        infoEl.className = 'pipe-info';
        const nameEl = document.createElement('div');
        nameEl.className = 'pipe-name';
        nameEl.textContent = names[p.pipeline] || p.pipeline;
        const barTrack = document.createElement('div');
        barTrack.className = 'pipe-bar-track';
        const barFill = document.createElement('div');
        barFill.className = 'pipe-bar-fill';
        barFill.style.background = barCol;
        barTrack.appendChild(barFill);
        infoEl.appendChild(nameEl);
        infoEl.appendChild(barTrack);

        const scoreEl = document.createElement('span');
        scoreEl.className = 'pipe-score';
        scoreEl.style.color = barCol;
        scoreEl.textContent = `${pct}%`;

        const timeEl = document.createElement('span');
        timeEl.className = 'pipe-time';
        timeEl.textContent = `${p.execution_ms}ms`;

        el.appendChild(iconEl);
        el.appendChild(infoEl);
        el.appendChild(scoreEl);
        el.appendChild(timeEl);
        list.appendChild(el);

        setTimeout(() => {
            barFill.style.width = `${pct}%`;
        }, 150 + i * 60);
    });

    // Report / Custody buttons
    document.getElementById('btnReport').onclick = () => {
        if (data.report_url) window.open(`${API.replace('/api/v1', '')}${data.report_url}`, '_blank');
    };
    document.getElementById('btnCustody').onclick = () => {
        loadCustody(data.analysis_id);
        switchView('forensic');
    };

    // Forensic viewer setup
    setupForensic(data);
}

function animateNum(el, from, to, dur) {
    const start = performance.now();
    (function tick(now) {
        const p = Math.min((now - start) / dur, 1);
        const e = 1 - Math.pow(1 - p, 4);
        el.textContent = (from + (to - from) * e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
    })(performance.now());
}

/* ═══ FORENSIC VIEWER ══════════════════════════════════════ */

// Track blob URLs so we can revoke them and avoid memory leaks
let _forensicBlobs = {};

function setupForensic(data) {
    // Normalise heatmap URL map: tab-key → relative API URL
    const heatmapMap = {};
    (data.heatmap_urls || []).forEach(u => {
        // gradcam tab gets either gradcam_efficientnet or gradcam_xception
        if (u.includes('gradcam_efficientnet')) heatmapMap['gradcam'] = u;
        else if (u.includes('gradcam_xception') && !heatmapMap['gradcam']) heatmapMap['gradcam'] = u;
        // ela tab
        if (u.includes('ela')) heatmapMap['ela'] = u;
        // copy_move tab
        if (u.includes('copy_move')) heatmapMap['copy_move'] = u;
    });

    // Revoke old blobs from previous analysis
    Object.values(_forensicBlobs).forEach(b => URL.revokeObjectURL(b));
    _forensicBlobs = {};

    const viewer = document.getElementById('viewer');

    // Helper — fetch image with auth and render as blob URL
    async function showAuthImage(apiRelUrl, altText) {
        viewer.innerHTML = '<div class="viewer-placeholder" style="animation:pulse 1s infinite">Loading…</div>';
        try {
            const fullUrl = `${window.location.origin}${apiRelUrl}`;
            const res = await authFetch(fullUrl);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            _forensicBlobs[altText] = blobUrl;

            viewer.innerHTML = '';
            const img = document.createElement('img');
            img.src = blobUrl;
            img.alt = altText;
            img.style.cssText = 'max-width:100%;max-height:500px;border-radius:8px;display:block;margin:auto';
            viewer.appendChild(img);
        } catch (e) {
            viewer.innerHTML = '';
            const p = document.createElement('div');
            p.className = 'viewer-placeholder';
            p.textContent = `Could not load ${altText}: ${e.message}`;
            viewer.appendChild(p);
        }
    }

    function showPlaceholder(text) {
        viewer.innerHTML = '';
        const p = document.createElement('div');
        p.className = 'viewer-placeholder';
        p.textContent = text;
        viewer.appendChild(p);
    }

    document.querySelectorAll('#fTabs .ftab').forEach(tab => {
        // Remove old listeners by cloning
        const fresh = tab.cloneNode(true);
        tab.parentNode.replaceChild(fresh, tab);
        fresh.addEventListener('click', () => {
            document.querySelectorAll('#fTabs .ftab').forEach(t => t.classList.remove('active'));
            fresh.classList.add('active');

            const key = fresh.dataset.tab;

            if (key === 'original') {
                if (data.analysis_id) {
                    showAuthImage(`/api/v1/results/${data.analysis_id}/original`, 'Original evidence file');
                } else {
                    showPlaceholder('No analysis loaded');
                }
            } else if (heatmapMap[key]) {
                showAuthImage(heatmapMap[key], key);
            } else {
                showPlaceholder(`${key.replace(/_/g, ' ')} — not generated for this file`);
            }
        });
    });

    // Auto-load original on analysis complete
    if (data.analysis_id) {
        showAuthImage(`/api/v1/results/${data.analysis_id}/original`, 'Original evidence file');
        // Mark original tab active
        document.querySelectorAll('#fTabs .ftab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === 'original');
        });
    }

    // EXIF
    renderExif(data.exif_data);
}


function renderExif(exif) {
    const el = document.getElementById('exifList');
    if (!exif || Object.keys(exif).length === 0) {
        el.innerHTML = '';
        const msg = document.createElement('div');
        msg.style.cssText = 'grid-column:1/-1; text-align:center; color:var(--text-500); padding:1.5rem';
        msg.textContent = 'No EXIF metadata — data may have been stripped';
        el.appendChild(msg);
        return;
    }

    const warnings = ['photoshop', 'gimp', 'faceapp', 'faceswap', 'deepfacelab'];
    const priority = ['Image Make', 'Image Model', 'EXIF DateTimeOriginal', 'Image Software', 'EXIF ExifImageWidth', 'EXIF ExifImageLength'];
    const sorted = [...new Set([...priority.filter(k => exif[k]), ...Object.keys(exif)])];

    el.innerHTML = '';
    sorted.slice(0, 16).forEach(key => {
        const val = String(exif[key]);
        const isAlert = warnings.some(w => val.toLowerCase().includes(w));

        const entry = document.createElement('div');
        entry.className = `exif-entry${isAlert ? ' alert' : ''}`;

        const keyEl = document.createElement('div');
        keyEl.className = 'exif-key';
        keyEl.textContent = key;

        const valEl = document.createElement('div');
        valEl.className = 'exif-val';
        valEl.textContent = val;  // textContent — NOT innerHTML (XSS prevention)

        entry.appendChild(keyEl);
        entry.appendChild(valEl);
        el.appendChild(entry);
    });
}

async function loadCustody(id) {
    const tl = document.getElementById('custodyTl');
    try {
        const res = await authFetch(`${API}/results/${id}/custody`);
        if (!res.ok) throw new Error('Failed');
        const logs = await res.json();
        if (!logs.length) {
            tl.innerHTML = '';
            const msg = document.createElement('div');
            msg.style.cssText = 'color:var(--text-500);padding:1rem';
            msg.textContent = 'No entries';
            tl.appendChild(msg);
            return;
        }

        const actionLabels = {
            evidence_intake: '📥 Evidence Intake',
            analysis_start: '🔬 Analysis Started',
            analysis_complete: '✅ Analysis Completed',
            analysis_failed: '❌ Analysis Failed',
        };

        tl.innerHTML = '';
        logs.forEach(l => {
            const item = document.createElement('div');
            item.className = 'tl-item';

            const timeEl = document.createElement('div');
            timeEl.className = 'tl-time';
            timeEl.textContent = l.timestamp;

            const actionEl = document.createElement('div');
            actionEl.className = 'tl-action';
            actionEl.textContent = actionLabels[l.action] || l.action;

            const detailEl = document.createElement('div');
            detailEl.className = 'tl-detail';
            detailEl.textContent = l.details;

            item.appendChild(timeEl);
            item.appendChild(actionEl);
            item.appendChild(detailEl);

            if (l.file_hash) {
                const hashEl = document.createElement('div');
                hashEl.className = 'tl-hash';
                hashEl.textContent = `SHA-256: ${l.file_hash.slice(0, 40)}…`;
                item.appendChild(hashEl);
            }

            tl.appendChild(item);
        });
    } catch {
        tl.innerHTML = '';
        const msg = document.createElement('div');
        msg.style.cssText = 'color:var(--text-500);padding:1rem';
        msg.textContent = 'Could not load custody log';
        tl.appendChild(msg);
    }
}

/* ═══ HISTORY ══════════════════════════════════════════════ */
async function loadHistory() {
    if (!Auth.isLoggedIn()) return;
    const tbody = document.getElementById('historyTbody');
    try {
        const res = await authFetch(`${API}/history?per_page=50`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data.analyses.length) {
            tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No analyses recorded</td></tr>';
            return;
        }
        tbody.innerHTML = '';
        data.analyses.forEach(a => {
            const tr = document.createElement('tr');
            tr.onclick = () => loadById(a.analysis_id);
            tr.style.cursor = 'pointer';

            const cells = [
                { html: false, text: a.evidence_id, cls: 'mono text-accent' },
                { html: false, text: a.filename.length > 28 ? a.filename.slice(0, 28) + '…' : a.filename },
                { html: false, text: a.media_type },
                { html: false, text: a.overall_score != null ? a.overall_score.toFixed(1) : '—', bold: true },
                { verdict: true, text: (a.verdict || '—').replace(/_/g, ' '), cls: a.verdict },
                { html: false, text: a.status },
                { html: false, text: new Date(a.created_at).toLocaleString(), cls: 'mono', style: 'color:var(--text-400)' },
            ];

            cells.forEach(c => {
                const td = document.createElement('td');
                if (c.verdict) {
                    const span = document.createElement('span');
                    span.className = `verdict-pill verdict-${c.cls}`;
                    span.style.cssText = 'padding:4px 12px; font-size:0.7rem';
                    span.textContent = c.text;
                    td.appendChild(span);
                } else if (c.bold) {
                    const strong = document.createElement('strong');
                    strong.textContent = c.text;
                    td.appendChild(strong);
                } else {
                    td.textContent = c.text;
                }
                if (c.cls && !c.verdict) td.className = c.cls;
                if (c.style) td.style.cssText = c.style;
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
    } catch { }
}

async function loadById(id) {
    try {
        const res = await authFetch(`${API}/results/${id}`);
        if (!res.ok) throw new Error();
        current = await res.json();
        renderResults(current);
        switchView('results');
    } catch { }
}

/* ═══ UTILITIES ════════════════════════════════════════════ */
function verdictColor(v) {
    return {
        AUTHENTIC: '#10b981', LIKELY_AUTHENTIC: '#84cc16', SUSPICIOUS: '#f59e0b',
        LIKELY_FAKE: '#f97316', CONFIRMED_FAKE: '#f43f5e'
    }[v] || '#64748b';
}

function scoreGradient(s) {
    if (s < 0.2) return '#10b981';
    if (s < 0.4) return '#84cc16';
    if (s < 0.6) return '#f59e0b';
    if (s < 0.8) return '#f97316';
    return '#f43f5e';
}

/* ═══ INIT ═════════════════════════════════════════════════ */
updateAuthUI();
if (Auth.isLoggedIn()) {
    loadHistory();
} else {
    showAuthModal();
}
