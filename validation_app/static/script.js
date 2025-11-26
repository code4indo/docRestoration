/**
 * Document Restoration Validation System - Frontend Script
 */

console.log("Validation System loaded");

// DOM Elements
let fileInput, uploadArea, previewContainer, previewImage, removeBtn;
let groundTruthInput, validateBtn;
let loadingOverlay, resultsSection;
let currentFile = null;
let currentResult = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize DOM elements
    fileInput = document.getElementById('file-input');
    uploadArea = document.getElementById('upload-area');
    previewContainer = document.getElementById('preview-container');
    previewImage = document.getElementById('preview-image');
    removeBtn = document.getElementById('remove-btn');
    groundTruthInput = document.getElementById('ground-truth');
    validateBtn = document.getElementById('validate-btn');
    loadingOverlay = document.getElementById('loading-overlay');
    resultsSection = document.getElementById('results-section');

    setupEventListeners();
});

function setupEventListeners() {
    // File input
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('click', () => {
        if (!currentFile) fileInput.click();
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Remove button
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeFile();
    });
    
    // Validate button
    validateBtn.addEventListener('click', runValidation);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.upload-content').style.display = 'none';
        previewContainer.style.display = 'block';
        validateBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeFile() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    document.querySelector('.upload-content').style.display = 'block';
    previewContainer.style.display = 'none';
    validateBtn.disabled = true;
}

async function runValidation() {
    if (!currentFile) {
        alert('Please upload an image first.');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('degraded_image', currentFile);
    formData.append('ground_truth', groundTruthInput.value.trim());
    
    try {
        // Update steps
        updateStep('step-restore', 'active');
        
        const response = await fetch('/api/validate', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            currentResult = result;
            displayResults(result);
        } else {
            throw new Error(result.error || 'Validation failed');
        }
        
    } catch (error) {
        console.error('Validation error:', error);
        alert(`Validation failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
    resultsSection.style.display = 'none';
    
    // Reset steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'done');
        step.querySelector('.step-icon').textContent = '⏳';
    });
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function updateStep(stepId, status) {
    const step = document.getElementById(stepId);
    if (!step) return;
    
    step.classList.remove('active', 'done');
    
    if (status === 'active') {
        step.classList.add('active');
        step.querySelector('.step-icon').textContent = '⏳';
    } else if (status === 'done') {
        step.classList.add('done');
        step.querySelector('.step-icon').textContent = '✓';
    }
}

function displayResults(result) {
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // --- Summary Cards ---
    const metrics = result.metrics;
    
    // CER Card - Using estimated ground truth from LLM
    if (result.estimated_ground_truth) {
        const cerImprovement = metrics.cer_improvement_percent || 0;
        document.getElementById('cer-improvement').textContent = 
            `${cerImprovement > 0 ? '+' : ''}${cerImprovement.toFixed(1)}%`;
        document.getElementById('cer-degraded').textContent = (metrics.cer_degraded || 0).toFixed(1);
        document.getElementById('cer-restored').textContent = (metrics.cer_restored || 0).toFixed(1);
        
        // Color based on improvement
        const cerCard = document.getElementById('cer-card');
        if (cerImprovement >= 25) {
            cerCard.style.borderColor = 'var(--secondary)';
        } else if (cerImprovement > 0) {
            cerCard.style.borderColor = 'var(--warning)';
        } else {
            cerCard.style.borderColor = 'var(--danger)';
        }
    } else {
        document.getElementById('cer-improvement').textContent = 'N/A';
        document.getElementById('cer-degraded').textContent = '--';
        document.getElementById('cer-restored').textContent = '--';
    }
    
    // LLM Analysis
    const llm = result.llm_analysis;
    
    if (llm && llm.status === 'success') {
        // Quality scores
        const degradedScore = llm.degraded_analysis?.overall_score || 0;
        const restoredScore = llm.restored_analysis?.overall_score || 0;
        
        document.getElementById('quality-degraded').textContent = degradedScore;
        document.getElementById('quality-restored').textContent = restoredScore;
        
        // Verdict - use Indonesian version if available
        const winner = llm.comparison?.winner || 'unknown';
        const verdictIcon = document.getElementById('verdict-icon');
        const verdictText = document.getElementById('verdict-text');
        const verdictDisplay = llm.verdict_indonesian || llm.verdict || '';
        
        if (winner === 'restored') {
            verdictIcon.textContent = '🏆';
            verdictText.innerHTML = `<span style="color: var(--restored-color)">Restorasi Berhasil</span><br>${verdictDisplay}`;
        } else if (winner === 'degraded') {
            verdictIcon.textContent = '⚠️';
            verdictText.innerHTML = `<span style="color: var(--warning)">Perlu Evaluasi</span><br>${verdictDisplay}`;
        } else {
            verdictIcon.textContent = '🤝';
            verdictText.innerHTML = `<span>Hasil Setara</span><br>${verdictDisplay}`;
        }
        
        // --- Score Bars ---
        // Degraded
        setScoreBar('degraded-readability', llm.degraded_analysis?.readability_score || 0);
        setScoreBar('degraded-completeness', llm.degraded_analysis?.completeness_score || 0);
        setScoreBar('degraded-accuracy', llm.degraded_analysis?.accuracy_score || 0);
        
        // Restored
        setScoreBar('restored-readability', llm.restored_analysis?.readability_score || 0);
        setScoreBar('restored-completeness', llm.restored_analysis?.completeness_score || 0);
        setScoreBar('restored-accuracy', llm.restored_analysis?.accuracy_score || 0);
        
        // Issues and improvements
        displayList('degraded-issues', llm.degraded_analysis?.issues || [], '⚠️');
        displayList('restored-improvements', llm.restored_analysis?.improvements || [], '✓');
        
        // LLM Analysis section - use Indonesian version if available
        document.getElementById('llm-verdict').textContent = llm.verdict_indonesian || llm.verdict || '--';
        document.getElementById('llm-recommendation').textContent = llm.recommendation_indonesian || llm.recommendation || '--';
        
        const confidence = llm.comparison?.confidence || 0;
        document.getElementById('llm-confidence').style.width = `${confidence}%`;
        document.getElementById('llm-confidence-val').textContent = `${confidence}%`;
        
    } else {
        // LLM failed
        document.getElementById('quality-degraded').textContent = '--';
        document.getElementById('quality-restored').textContent = '--';
        document.getElementById('verdict-text').textContent = llm?.error || 'LLM analysis unavailable';
    }
    
    // --- Estimated Ground Truth Section ---
    displayEstimatedGroundTruth(result);
    
    // --- Images ---
    document.getElementById('degraded-image').src = result.degraded.image_url;
    document.getElementById('restored-image').src = result.restored.image_url;
    
    // --- Text ---
    displayText('degraded-text', result.degraded.htr_text);
    displayText('restored-text', result.restored.htr_text);
    
    document.getElementById('degraded-lines').textContent = `${result.degraded.line_count || 0} baris`;
    document.getElementById('restored-lines').textContent = `${result.restored.line_count || 0} baris`;
}

function displayEstimatedGroundTruth(result) {
    const gtDutchContainer = document.getElementById('estimated-gt-text');
    const gtIndonesianContainer = document.getElementById('estimated-gt-indonesian');
    const gtConfidence = document.getElementById('gt-confidence');
    const gtConfidenceLevel = document.getElementById('gt-confidence-level');
    const cerVsGtDegraded = document.getElementById('cer-vs-gt-degraded');
    const cerVsGtRestored = document.getElementById('cer-vs-gt-restored');
    
    if (!gtDutchContainer) return;
    
    const estimatedGt = result.estimated_ground_truth;
    const estimatedGtIndonesian = result.llm_analysis?.estimated_ground_truth_indonesian;
    const llm = result.llm_analysis;
    const metrics = result.metrics;
    
    // Display document context if available
    const docContextCard = document.getElementById('doc-context-card');
    const docContextText = document.getElementById('document-context-text');
    if (llm?.document_context && docContextCard && docContextText) {
        // Show Indonesian context if available, otherwise Dutch
        const contextText = llm.document_context_id || llm.document_context;
        docContextText.textContent = contextText;
        docContextCard.style.display = 'block';
    } else if (docContextCard) {
        docContextCard.style.display = 'none';
    }
    
    // Display confidence explanation if available
    const confidenceCard = document.getElementById('confidence-explanation-card');
    const confidenceText = document.getElementById('confidence-explanation-text');
    if (llm?.confidence_explanation && confidenceCard && confidenceText) {
        confidenceText.textContent = llm.confidence_explanation;
        confidenceCard.style.display = 'block';
    } else if (confidenceCard) {
        confidenceCard.style.display = 'none';
    }
    
    // Display reconstruction notes if available
    const notesCard = document.getElementById('reconstruction-notes');
    const notesText = document.getElementById('reconstruction-notes-text');
    if (llm?.reconstruction_notes && notesCard && notesText) {
        notesText.textContent = llm.reconstruction_notes;
        notesCard.style.display = 'block';
    } else if (notesCard) {
        notesCard.style.display = 'none';
    }
    
    // Display Dutch GT
    if (estimatedGt && estimatedGt.trim()) {
        gtDutchContainer.innerHTML = '';
        const lines = estimatedGt.split('\n');
        lines.forEach((line, idx) => {
            if (line.trim()) {
                const lineDiv = document.createElement('div');
                lineDiv.className = 'gt-line';
                lineDiv.innerHTML = `<span class="gt-line-num">${idx + 1}.</span> ${escapeHtml(line)}`;
                gtDutchContainer.appendChild(lineDiv);
            }
        });
    } else {
        gtDutchContainer.innerHTML = '<p class="no-text">Ground truth estimasi tidak tersedia</p>';
    }
    
    // Display Indonesian Translation
    if (gtIndonesianContainer) {
        if (estimatedGtIndonesian && estimatedGtIndonesian.trim()) {
            gtIndonesianContainer.innerHTML = '';
            const lines = estimatedGtIndonesian.split('\n');
            lines.forEach((line, idx) => {
                if (line.trim()) {
                    const lineDiv = document.createElement('div');
                    lineDiv.className = 'gt-line';
                    lineDiv.innerHTML = `<span class="gt-line-num">${idx + 1}.</span> ${escapeHtml(line)}`;
                    gtIndonesianContainer.appendChild(lineDiv);
                }
            });
        } else {
            gtIndonesianContainer.innerHTML = '<p class="no-text">Terjemahan Indonesia tidak tersedia</p>';
        }
    }
    
    // Show confidence with level
    const confidence = llm?.comparison?.confidence || 0;
    const confidenceLevel = llm?.comparison?.confidence_level || getConfidenceLevel(confidence);
    
    gtConfidence.textContent = `Confidence: ${confidence}%`;
    
    if (gtConfidenceLevel) {
        gtConfidenceLevel.textContent = confidenceLevel.replace('_', ' ');
        gtConfidenceLevel.className = `gt-confidence-level ${confidenceLevel}`;
    }
    
    // Show CER vs GT
    const cerDegraded = metrics.cer_degraded || llm?.degraded_analysis?.cer_estimated * 100 || 0;
    const cerRestored = metrics.cer_restored || llm?.restored_analysis?.cer_estimated * 100 || 0;
    
    cerVsGtDegraded.textContent = `${cerDegraded.toFixed(1)}%`;
    cerVsGtRestored.textContent = `${cerRestored.toFixed(1)}%`;
    
    // Highlight improvement
    if (cerRestored < cerDegraded) {
        cerVsGtRestored.classList.add('highlight');
    }
}

// Helper function to determine confidence level from score
function getConfidenceLevel(confidence) {
    if (confidence >= 90) return 'very_high';
    if (confidence >= 75) return 'high';
    if (confidence >= 50) return 'medium';
    if (confidence >= 25) return 'low';
    return 'very_low';
}

// Tab switching function
function switchGtTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.gt-tab').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tab) {
            btn.classList.add('active');
        }
    });
    
    // Update content
    const dutchContent = document.getElementById('gt-dutch-content');
    const indonesianContent = document.getElementById('gt-indonesian-content');
    
    if (tab === 'dutch') {
        dutchContent.style.display = 'block';
        dutchContent.classList.add('active');
        indonesianContent.style.display = 'none';
        indonesianContent.classList.remove('active');
    } else {
        dutchContent.style.display = 'none';
        dutchContent.classList.remove('active');
        indonesianContent.style.display = 'block';
        indonesianContent.classList.add('active');
    }
}

function setScoreBar(id, value) {
    const bar = document.getElementById(id);
    const valEl = document.getElementById(`${id}-val`);
    
    if (bar) {
        bar.style.width = `${value}%`;
    }
    if (valEl) {
        valEl.textContent = value;
    }
}

function displayList(containerId, items, prefix) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (items.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    const ul = document.createElement('ul');
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${prefix} ${item}`;
        ul.appendChild(li);
    });
    
    container.innerHTML = '';
    container.appendChild(ul);
}

function displayText(containerId, text) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (text && text.trim()) {
        container.innerHTML = '';
        
        // Split into lines and add line numbers
        const lines = text.split('\n');
        lines.forEach((line, idx) => {
            if (line.trim()) {  // Only show non-empty lines
                const div = document.createElement('div');
                div.className = 'text-line';
                div.innerHTML = `<span class="line-num">${idx + 1}.</span>${escapeHtml(line)}`;
                container.appendChild(div);
            }
        });
        
        // If no lines shown
        if (container.children.length === 0) {
            container.innerHTML = '<p class="no-text">Tidak ada teks terdeteksi</p>';
        }
    } else {
        container.innerHTML = '<p class="no-text">Tidak ada teks terdeteksi</p>';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Modal functions
function openImageModal(type) {
    const modal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    
    if (type === 'degraded') {
        modalImage.src = document.getElementById('degraded-image').src;
    } else {
        modalImage.src = document.getElementById('restored-image').src;
    }
    
    modal.classList.add('active');
}

function closeImageModal() {
    document.getElementById('image-modal').classList.remove('active');
}

// Close modal on click outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('image-modal');
    if (e.target === modal) {
        closeImageModal();
    }
});

// Reset validation
function resetValidation() {
    removeFile();
    groundTruthInput.value = '';
    resultsSection.style.display = 'none';
    currentResult = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Download report
function downloadReport() {
    if (!currentResult) {
        alert('No results to download.');
        return;
    }
    
    const report = generateReport(currentResult);
    const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `validation_report_${currentResult.session_id}.txt`;
    a.click();
    
    URL.revokeObjectURL(url);
}

function generateReport(result) {
    const lines = [
        '═'.repeat(60),
        '  DOCUMENT RESTORATION VALIDATION REPORT',
        '═'.repeat(60),
        '',
        `Session ID: ${result.session_id}`,
        `Timestamp: ${result.timestamp}`,
        '',
        '─'.repeat(60),
        '  CER METRICS (vs LLM Estimated Ground Truth)',
        '─'.repeat(60),
    ];
    
    if (result.estimated_ground_truth) {
        const metrics = result.metrics;
        lines.push(`CER Degraded: ${metrics.cer_degraded || '--'}%`);
        lines.push(`CER Restored: ${metrics.cer_restored || '--'}%`);
        lines.push(`Improvement: ${metrics.cer_improvement_percent || 0}%`);
        
        if ((metrics.cer_improvement_percent || 0) >= 25) {
            lines.push('');
            lines.push('✓ TARGET MET: CER improvement ≥ 25%');
        }
    } else {
        lines.push('(No estimated ground truth available)');
    }
    
    // Estimated Ground Truth
    lines.push('');
    lines.push('─'.repeat(60));
    lines.push('  ESTIMATED GROUND TRUTH (LLM Generated)');
    lines.push('─'.repeat(60));
    lines.push(result.estimated_ground_truth || '(Not available)');
    
    lines.push('');
    lines.push('─'.repeat(60));
    lines.push('  HTR RESULTS - DEGRADED IMAGE');
    lines.push('─'.repeat(60));
    lines.push(result.degraded.htr_text || '(No text detected)');
    
    lines.push('');
    lines.push('─'.repeat(60));
    lines.push('  HTR RESULTS - RESTORED IMAGE');
    lines.push('─'.repeat(60));
    lines.push(result.restored.htr_text || '(No text detected)');
    
    if (result.llm_analysis && result.llm_analysis.status === 'success') {
        lines.push('');
        lines.push('─'.repeat(60));
        lines.push('  LLM ANALYSIS (Grok 4.1 Fast)');
        lines.push('─'.repeat(60));
        lines.push('');
        
        if (result.llm_analysis.reconstruction_notes) {
            lines.push('Reconstruction Notes:');
            lines.push(result.llm_analysis.reconstruction_notes);
            lines.push('');
        }
        
        lines.push('Verdict:');
        lines.push(result.llm_analysis.verdict || '--');
        lines.push('');
        lines.push('Recommendation:');
        lines.push(result.llm_analysis.recommendation || '--');
        lines.push('');
        lines.push(`Confidence: ${result.llm_analysis.comparison?.confidence || 0}%`);
        lines.push(`Winner: ${result.llm_analysis.comparison?.winner || 'unknown'}`);
    }
    
    lines.push('');
    lines.push('═'.repeat(60));
    lines.push('  END OF REPORT');
    lines.push('═'.repeat(60));
    
    return lines.join('\n');
}
