document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-menu a');
    const sections = document.querySelectorAll('.section-container');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            sections.forEach(section => {
                if (section.id === targetId) {
                    section.classList.add('active-section');
                } else {
                    section.classList.remove('active-section');
                }
            });
        });
    });

    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropzoneContent = document.querySelector('.dropzone-content');
    const classifyBtn = document.getElementById('classify-btn');
    const cameraBtn = document.getElementById('camera-btn');
    const changeImgBtn = document.getElementById('change-img-btn');
    const cameraStream = document.getElementById('camera-stream');
    const captureBtn = document.getElementById('capture-btn');

    const resultCard = document.getElementById('result-card');
    const resultPlaceholder = document.querySelector('.result-placeholder');
    const resultContent = document.querySelector('.result-content');
    const resultLoader = document.getElementById('result-loader');

    const roastBadge = document.getElementById('roast-badge');
    const roastTitle = document.getElementById('roast-title');
    const confidenceVal = document.getElementById('confidence-val');
    const confidenceBar = document.getElementById('confidence-bar');
    const roastDesc = document.getElementById('roast-desc');
    const roastFlavor = document.getElementById('roast-flavor');
    const roastBrew = document.getElementById('roast-brew');
    const mockWarning = document.getElementById('mock-warning');

    let selectedFile = null;
    let localStream = null;

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('File harus berupa gambar!');
            return;
        }
        selectedFile = file;
        
        // Hide camera stream if running
        stopCamera();

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropzoneContent.style.display = 'none';
            classifyBtn.disabled = false;
            
            // Tampilkan tombol Ganti Gambar, sembunyikan tombol Kamera
            changeImgBtn.classList.remove('hidden');
            cameraBtn.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Event Listener Ganti Gambar
    changeImgBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        dropzoneContent.style.display = 'block';
        classifyBtn.disabled = true;
        
        changeImgBtn.classList.add('hidden');
        cameraBtn.classList.remove('hidden');
        
        // Reset card hasil kembali kosong
        resultContent.classList.add('hidden');
        resultLoader.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
        resultCard.classList.add('empty');
    });

    // Camera Integration
    cameraBtn.addEventListener('click', async () => {
        if (localStream) {
            stopCamera();
            return;
        }

        try {
            const constraints = {
                video: { facingMode: 'environment' } // Prefer back camera on mobile
            };
            localStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraStream.srcObject = localStream;
            cameraStream.style.display = 'block';
            dropzone.style.display = 'none';
            
            cameraBtn.innerHTML = '<i class="fa-solid fa-xmark"></i> Tutup Kamera';
            captureBtn.classList.remove('hidden');
            classifyBtn.disabled = true;
        } catch (err) {
            console.error('Kamera gagal diakses:', err);
            alert('Gagal mengakses kamera. Silakan pilih opsi unggah manual.');
        }
    });

    captureBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = cameraStream.videoWidth || 640;
        canvas.height = cameraStream.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], 'captured_image.png', { type: 'image/png' });
            handleFile(file);
        }, 'image/png');
    });

    function stopCamera() {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        cameraStream.style.display = 'none';
        dropzone.style.display = 'flex';
        cameraBtn.innerHTML = '<i class="fa-solid fa-camera"></i> Gunakan Kamera';
        captureBtn.classList.add('hidden');
    }

    // Classify Action
    classifyBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show Loader
        resultPlaceholder.classList.add('hidden');
        resultContent.classList.add('hidden');
        resultLoader.classList.remove('hidden');
        resultCard.classList.remove('empty');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errorMsg = errorData.detail || errorData.error || 'Gagal memproses gambar';
                throw new Error(errorMsg);
            }

            const data = await response.json();
            
            // Render Result
            roastBadge.innerText = `${data.class} Roast`;
            roastTitle.innerText = data.title;
            confidenceVal.innerText = `${data.confidence}%`;
            confidenceBar.style.width = `${data.confidence}%`;
            roastDesc.innerText = data.description;
            roastFlavor.innerText = data.flavor_notes;
            roastBrew.innerText = data.brew_recommendation;

            // Change Badge color
            setThemeForClass(data.class);

            if (data.is_mock) {
                mockWarning.classList.remove('hidden');
            } else {
                mockWarning.classList.add('hidden');
            }

            resultLoader.classList.add('hidden');
            resultContent.classList.remove('hidden');
            
            // Gulir layar secara halus ke bagian kartu hasil (auto-scroll)
            setTimeout(() => {
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
            
        } catch (error) {
            console.error(error);
            alert('Klasifikasi Gagal:\n' + error.message);
            resultLoader.classList.add('hidden');
            resultPlaceholder.classList.remove('hidden');
            resultCard.classList.add('empty');
        }
    });

    function setThemeForClass(className) {
        // Reset colors
        roastBadge.style.backgroundColor = '';
        roastBadge.style.color = '';

        if (className === 'Green') {
            roastBadge.style.backgroundColor = '#81C784';
            roastBadge.style.color = '#1B5E20';
        } else if (className === 'Light') {
            roastBadge.style.backgroundColor = '#FFD54F';
            roastBadge.style.color = '#E65100';
        } else if (className === 'Medium') {
            roastBadge.style.backgroundColor = '#A1887F';
            roastBadge.style.color = '#FFF';
        } else if (className === 'Dark') {
            roastBadge.style.backgroundColor = '#3E2723';
            roastBadge.style.color = '#FFF';
        }
    }

    // Load Stats
    async function loadStats() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    document.getElementById('stat-total').innerText = data.total_images.toLocaleString();
                    document.getElementById('stat-train').innerText = data.splits.train.toLocaleString();
                    document.getElementById('stat-test').innerText = data.splits.test.toLocaleString();

                    // Render custom charts
                    const classChart = document.getElementById('class-chart');
                    classChart.innerHTML = '';

                    const maxVal = Math.max(...Object.values(data.distribution));

                    for (const [label, count] of Object.entries(data.distribution)) {
                        const pct = (count / maxVal) * 100;
                        const row = document.createElement('div');
                        row.className = 'chart-row';
                        row.innerHTML = `
                            <div class="chart-label">${label}</div>
                            <div class="chart-bar-container">
                                <div class="chart-bar" style="width: ${pct}%; background-color: ${getColorForLabel(label)}"></div>
                            </div>
                            <div class="chart-val">${count} pcs</div>
                        `;
                        classChart.appendChild(row);
                    }
                }
            }
        } catch (err) {
            console.error('Gagal mengambil data statistik:', err);
        }
    }

    function getColorForLabel(label) {
        if (label === 'Dark') return '#3E2723';
        if (label === 'Green') return '#81C784';
        if (label === 'Light') return '#FFD54F';
        return '#A1887F'; // Medium
    }

    // Quality Evaluator Form Controls & Submission
    const qualityForm = document.getElementById('quality-form');
    const inputDefect = document.getElementById('input-defect');
    const inputDiameter = document.getElementById('input-diameter');
    const inputMoisture = document.getElementById('input-moisture');

    const valDefect = document.getElementById('val-defect');
    const valDiameter = document.getElementById('val-diameter');
    const valMoisture = document.getElementById('val-moisture');

    const qualityResultCard = document.getElementById('quality-result-card');
    const qualityPlaceholder = document.getElementById('quality-placeholder');
    const qualityContent = document.getElementById('quality-content');
    const qualityLoader = document.getElementById('quality-loader');

    const qualityBadge = document.getElementById('quality-badge');
    const qualityTitle = document.getElementById('quality-title');
    const qualityDesc = document.getElementById('quality-desc');
    const qualityReasons = document.getElementById('quality-reasons');
    const qualityRec = document.getElementById('quality-rec');

    // Update range labels in real-time
    inputDefect.addEventListener('input', () => {
        valDefect.innerText = `${inputDefect.value}%`;
    });
    inputDiameter.addEventListener('input', () => {
        valDiameter.innerText = `${inputDiameter.value} mm`;
    });
    inputMoisture.addEventListener('input', () => {
        valMoisture.innerText = `${inputMoisture.value}%`;
    });

    // Form submit handler
    qualityForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show Loader
        qualityPlaceholder.classList.add('hidden');
        qualityContent.classList.add('hidden');
        qualityLoader.classList.remove('hidden');
        qualityResultCard.classList.remove('empty');

        const payload = {
            defect_percentage: parseFloat(inputDefect.value),
            average_diameter: parseFloat(inputDiameter.value),
            moisture_content: parseFloat(inputMoisture.value)
        };

        try {
            const response = await fetch('/api/quality-evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Gagal mengevaluasi kualitas');
            }

            const data = await response.json();

            // Render result
            qualityBadge.innerText = data.quality_class;
            qualityTitle.innerText = data.title;
            qualityDesc.innerText = data.description;
            qualityRec.innerText = data.recommendation;

            // Render reasons list
            qualityReasons.innerHTML = '';
            data.reasons.forEach(reason => {
                const li = document.createElement('li');
                li.innerText = reason;
                qualityReasons.appendChild(li);
            });

            // Set badge theme
            if (data.quality_class === 'Kualitas Ekspor') {
                qualityBadge.style.backgroundColor = '#D4AF37'; // Gold
                qualityBadge.style.color = '#15100E';
            } else {
                qualityBadge.style.backgroundColor = '#A1887F'; // Warm brown
                qualityBadge.style.color = '#FFF';
            }

            qualityLoader.classList.add('hidden');
            qualityContent.classList.remove('hidden');
        } catch (error) {
            console.error(error);
            alert('Terjadi kesalahan saat evaluasi kualitas: ' + error.message);
            qualityLoader.classList.add('hidden');
            qualityPlaceholder.classList.remove('hidden');
            qualityResultCard.classList.add('empty');
        }
    });

    loadStats();
});
