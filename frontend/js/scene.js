/**
 * SecureSight — 3D Background Scene (Three.js)
 *
 * Animated particle network with a rotating wireframe icosahedron (shield motif).
 * Particles drift slowly and connect with lines when close enough.
 * GPU-friendly: uses BufferGeometry + Points for max performance.
 */

(function () {
    'use strict';

    const container = document.getElementById('scene-container');
    if (!container) return;

    // ── Renderer ─────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 2000);
    camera.position.set(0, 0, 500);

    // ── Colors ───────────────────────────────────────────────
    const ACCENT = new THREE.Color(0x00ddb3);
    const PURPLE = new THREE.Color(0x7c5dfa);
    const DIM = new THREE.Color(0x1e293b);

    // ── Wireframe Icosahedron (Shield) ───────────────────────
    const icoGeo = new THREE.IcosahedronGeometry(120, 1);
    const icoMat = new THREE.MeshBasicMaterial({
        color: ACCENT,
        wireframe: true,
        transparent: true,
        opacity: 0.06,
    });
    const ico = new THREE.Mesh(icoGeo, icoMat);
    scene.add(ico);

    // Inner glow sphere
    const glowGeo = new THREE.IcosahedronGeometry(80, 2);
    const glowMat = new THREE.MeshBasicMaterial({
        color: PURPLE,
        wireframe: true,
        transparent: true,
        opacity: 0.03,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    scene.add(glow);

    // ── Particles ────────────────────────────────────────────
    const PARTICLE_COUNT = 180;
    const SPREAD = 900;
    const CONNECT_DIST = 140;

    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const velocities = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;
        positions[i3] = (Math.random() - 0.5) * SPREAD;
        positions[i3 + 1] = (Math.random() - 0.5) * SPREAD;
        positions[i3 + 2] = (Math.random() - 0.5) * SPREAD * 0.5;

        velocities[i3] = (Math.random() - 0.5) * 0.3;
        velocities[i3 + 1] = (Math.random() - 0.5) * 0.3;
        velocities[i3 + 2] = (Math.random() - 0.5) * 0.15;

        // Blend between accent and purple
        const t = Math.random();
        const c = ACCENT.clone().lerp(PURPLE, t);
        colors[i3] = c.r;
        colors[i3 + 1] = c.g;
        colors[i3 + 2] = c.b;
    }

    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const pMat = new THREE.PointsMaterial({
        size: 2.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    const particles = new THREE.Points(pGeo, pMat);
    scene.add(particles);

    // ── Connection Lines ─────────────────────────────────────
    const MAX_LINES = 300;
    const linePositions = new Float32Array(MAX_LINES * 6);
    const lineColors = new Float32Array(MAX_LINES * 6);

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    lineGeo.setDrawRange(0, 0);

    const lineMat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    // ── Mouse interaction ────────────────────────────────────
    let mouseX = 0, mouseY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    // ── Resize ───────────────────────────────────────────────
    function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    window.addEventListener('resize', onResize);

    // ── Animation Loop ───────────────────────────────────────
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);

        const t = clock.getElapsedTime();
        const dt = clock.getDelta();

        // Rotate wireframe shield
        ico.rotation.x = t * 0.05;
        ico.rotation.y = t * 0.08;
        ico.rotation.z = t * 0.03;

        glow.rotation.x = -t * 0.04;
        glow.rotation.y = -t * 0.06;

        // Subtle breathing
        const breathe = 1.0 + Math.sin(t * 0.5) * 0.03;
        ico.scale.setScalar(breathe);

        // Move particles
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const i3 = i * 3;
            positions[i3] += velocities[i3];
            positions[i3 + 1] += velocities[i3 + 1];
            positions[i3 + 2] += velocities[i3 + 2];

            // Bounce within bounds
            if (Math.abs(positions[i3]) > SPREAD * 0.5) velocities[i3] *= -1;
            if (Math.abs(positions[i3 + 1]) > SPREAD * 0.5) velocities[i3 + 1] *= -1;
            if (Math.abs(positions[i3 + 2]) > SPREAD * 0.3) velocities[i3 + 2] *= -1;
        }

        pGeo.attributes.position.needsUpdate = true;

        // Update connection lines
        let lineIndex = 0;

        for (let i = 0; i < PARTICLE_COUNT && lineIndex < MAX_LINES; i++) {
            for (let j = i + 1; j < PARTICLE_COUNT && lineIndex < MAX_LINES; j++) {
                const i3 = i * 3;
                const j3 = j * 3;

                const dx = positions[i3] - positions[j3];
                const dy = positions[i3 + 1] - positions[j3 + 1];
                const dz = positions[i3 + 2] - positions[j3 + 2];
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < CONNECT_DIST) {
                    const li = lineIndex * 6;
                    const alpha = 1 - dist / CONNECT_DIST;

                    linePositions[li] = positions[i3];
                    linePositions[li + 1] = positions[i3 + 1];
                    linePositions[li + 2] = positions[i3 + 2];
                    linePositions[li + 3] = positions[j3];
                    linePositions[li + 4] = positions[j3 + 1];
                    linePositions[li + 5] = positions[j3 + 2];

                    const c1 = ACCENT.clone().lerp(DIM, 1 - alpha);
                    lineColors[li] = c1.r; lineColors[li + 1] = c1.g; lineColors[li + 2] = c1.b;
                    lineColors[li + 3] = c1.r; lineColors[li + 4] = c1.g; lineColors[li + 5] = c1.b;

                    lineIndex++;
                }
            }
        }

        lineGeo.setDrawRange(0, lineIndex * 2);
        lineGeo.attributes.position.needsUpdate = true;
        lineGeo.attributes.color.needsUpdate = true;

        // Camera follows mouse subtly
        camera.position.x += (mouseX * 40 - camera.position.x) * 0.02;
        camera.position.y += (-mouseY * 30 - camera.position.y) * 0.02;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }

    animate();
})();
