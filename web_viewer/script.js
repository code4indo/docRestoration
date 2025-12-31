/**
 * GAN-HTR Document Viewer - Script
 * 
 * Standalone viewer for document restoration + HTR results
 * Auto-loads results from server API - no manual folder selection needed
 */

console.log("GAN-HTR Document Viewer loaded");

// Global state
let documentsData = {};
let currentObjectUrls = { image: null };
let currentZoom = 1.0;
let showBoxes = true;
let autoRefreshInterval = null;
let lastKnownTimestamp = 0;

// Pan/drag state
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let scrollStartX = 0;
let scrollStartY = 0;

// API Base URL (same origin)
const API_BASE = '';

// DOM Elements (initialized after DOMContentLoaded)
let documentList, imageContainer, textContent;
let zoomInBtn, zoomOutBtn, zoomResetBtn, zoomLevelDisplay;
let toggleBoxesBtn, exportTextBtn, copyTextBtn, lineCountDisplay;
let refreshBtn, autoRefreshToggle, statusIndicator;

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM ready. Initializing...");
    
    // Get DOM elements
    documentList = document.getElementById('document-list');
    imageContainer = document.getElementById('image-container');
    textContent = document.getElementById('text-content');
    zoomInBtn = document.getElementById('zoom-in');
    zoomOutBtn = document.getElementById('zoom-out');
    zoomResetBtn = document.getElementById('zoom-reset');
    zoomLevelDisplay = document.getElementById('zoom-level');
    toggleBoxesBtn = document.getElementById('toggle-boxes');
    exportTextBtn = document.getElementById('export-text');
    copyTextBtn = document.getElementById('copy-text');
    lineCountDisplay = document.getElementById('line-count');
    refreshBtn = document.getElementById('refresh-btn');
    autoRefreshToggle = document.getElementById('auto-refresh');
    statusIndicator = document.getElementById('status-indicator');

    // Setup event listeners
    setupEventListeners();
    
    // Initialize Split.js for resizable panels
    if (typeof Split !== 'undefined') {
        Split(['#sidebar-panel', '#image-panel', '#text-panel'], {
            sizes: [15, 55, 30],
            minSize: [150, 300, 200],
            gutterSize: 6,
            cursor: 'col-resize'
        });
    }
    
    // Auto-load results from API
    loadResultsFromAPI();
    
    // Start auto-refresh (check every 5 seconds)
    startAutoRefresh();
    
    console.log("Initialization complete. Auto-loading results...");
});

function setupEventListeners() {
    // Refresh button
    if (refreshBtn) refreshBtn.addEventListener('click', loadResultsFromAPI);
    
    // Auto-refresh toggle
    if (autoRefreshToggle) {
        autoRefreshToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });
    }
    
    // Zoom controls
    if (zoomInBtn) zoomInBtn.addEventListener('click', () => setZoom(currentZoom + 0.25));
    if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => setZoom(currentZoom - 0.25));
    if (zoomResetBtn) zoomResetBtn.addEventListener('click', () => setZoom(1.0));
    
    // Mouse wheel zoom on image container
    if (imageContainer) {
        imageContainer.addEventListener('wheel', handleMouseWheelZoom, { passive: false });
        
        // Drag to pan
        imageContainer.addEventListener('mousedown', handlePanStart);
        imageContainer.addEventListener('mousemove', handlePanMove);
        imageContainer.addEventListener('mouseup', handlePanEnd);
        imageContainer.addEventListener('mouseleave', handlePanEnd);
    }
    
    // Toolbar actions
    if (toggleBoxesBtn) toggleBoxesBtn.addEventListener('click', toggleBoundingBoxes);
    if (exportTextBtn) exportTextBtn.addEventListener('click', exportAllText);
    if (copyTextBtn) copyTextBtn.addEventListener('click', copyCurrentText);
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        stopAutoRefresh();
        if (currentObjectUrls.image) {
            URL.revokeObjectURL(currentObjectUrls.image);
        }
    });
}

function startAutoRefresh() {
    if (autoRefreshInterval) return;
    autoRefreshInterval = setInterval(checkForNewResults, 5000);
    console.log("Auto-refresh started (5s interval)");
    updateStatusIndicator('connected');
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        console.log("Auto-refresh stopped");
        updateStatusIndicator('paused');
    }
}

function updateStatusIndicator(status) {
    if (!statusIndicator) return;
    statusIndicator.className = 'status-indicator ' + status;
    statusIndicator.title = status === 'connected' ? 'Auto-refresh aktif' : 
                           status === 'paused' ? 'Auto-refresh nonaktif' : 
                           'Memuat...';
}

async function checkForNewResults() {
    try {
        const response = await fetch(`${API_BASE}/api/latest`);
        if (!response.ok) return;
        
        const data = await response.json();
        if (data.latest && data.latest.timestamp > lastKnownTimestamp) {
            console.log("New result detected! Reloading...");
            lastKnownTimestamp = data.latest.timestamp;
            await loadResultsFromAPI();
            
            // Auto-select the latest document
            if (data.latest.id) {
                loadDocument(data.latest.id);
            }
        }
    } catch (err) {
        console.warn("Auto-refresh check failed:", err);
    }
}

async function loadResultsFromAPI() {
    console.log("Loading results from API...");
    updateStatusIndicator('loading');
    
    try {
        const response = await fetch(`${API_BASE}/api/results`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`Found ${data.count} results`);
        
        // Store data
        documentsData = {};
        data.results.forEach(doc => {
            documentsData[doc.id] = {
                id: doc.id,
                name: doc.name,
                imagePath: doc.image,
                xmlPath: doc.xml,
                timestamp: doc.timestamp
            };
            
            // Track latest timestamp
            if (doc.timestamp > lastKnownTimestamp) {
                lastKnownTimestamp = doc.timestamp;
            }
        });
        
        // Update document list
        loadDocumentList(data.results);
        updateStatusIndicator('connected');
        
    } catch (err) {
        console.error("Failed to load results:", err);
        if (documentList) {
            documentList.innerHTML = `<p class="placeholder">Gagal memuat hasil.<br><small>${err.message}</small></p>`;
        }
        updateStatusIndicator('error');
    }
}

function loadDocumentList(documents = []) {
    if (!documentList) return;

    documentList.innerHTML = '';

    if (documents.length === 0) {
        documentList.innerHTML = '<p class="placeholder">Belum ada hasil.<br><small>Proses dokumen di Gradio, hasil akan otomatis muncul di sini.</small></p>';
        return;
    }

    documents.forEach((doc, index) => {
        const item = document.createElement('div');
        item.className = 'doc-item';
        item.dataset.docId = doc.id;
        
        // Format timestamp
        const date = new Date(doc.timestamp * 1000);
        const timeStr = date.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
        
        item.innerHTML = `
            <div class="doc-name">${doc.name}</div>
            <div class="doc-meta">${timeStr}</div>
        `;
        item.addEventListener('click', () => loadDocument(doc.id));
        documentList.appendChild(item);
        
        // Mark newest with indicator
        if (index === 0) {
            item.classList.add('newest');
        }
    });

    // Auto-load first document if none selected
    if (documents.length > 0 && !documentList.querySelector('.active')) {
        loadDocument(documents[0].id);
    }
}

async function loadDocument(docId) {
    console.log(`Loading document: ${docId}`);

    // Update active state in list
    documentList.querySelectorAll('.doc-item').forEach(el => el.classList.remove('active'));
    const activeItem = documentList.querySelector(`[data-doc-id="${docId}"]`);
    if (activeItem) activeItem.classList.add('active');

    // Show loading state
    imageContainer.innerHTML = '<p class="placeholder">Memuat gambar...</p>';
    textContent.innerHTML = '<p class="placeholder">Memuat teks...</p>';

    // Cleanup previous
    if (currentObjectUrls.image) {
        URL.revokeObjectURL(currentObjectUrls.image);
        currentObjectUrls.image = null;
    }
    removeHighlight();
    const oldLayer = document.getElementById('highlight-layer');
    if (oldLayer) oldLayer.remove();

    const docData = documentsData[docId];
    if (!docData || !docData.imagePath || !docData.xmlPath) {
        console.error(`Incomplete data for document ${docId}`);
        imageContainer.innerHTML = '<p class="placeholder">Error: Data tidak lengkap.</p>';
        textContent.innerHTML = '<p class="placeholder">Error: Data tidak lengkap.</p>';
        return;
    }

    try {
        // Fetch XML from server
        const xmlResponse = await fetch(`${API_BASE}/${docData.xmlPath}`);
        if (!xmlResponse.ok) throw new Error(`Failed to load XML: ${xmlResponse.status}`);
        const xmlText = await xmlResponse.text();
        
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, "text/xml");
        
        // Get page dimensions
        const pageElement = xmlDoc.getElementsByTagName('Page')[0];
        const originalWidth = pageElement ? parseInt(pageElement.getAttribute('imageWidth')) : null;
        const originalHeight = pageElement ? parseInt(pageElement.getAttribute('imageHeight')) : null;

        // Create image (direct URL from server)
        const imageUrl = `${API_BASE}/${docData.imagePath}`;

        const img = document.createElement('img');
        img.alt = `Document ${docId}`;
        
        await new Promise((resolve, reject) => {
            img.onload = () => {
                imageContainer.innerHTML = '';
                imageContainer.appendChild(img);
                
                if (originalWidth) img.dataset.originalWidth = originalWidth;
                if (originalHeight) img.dataset.originalHeight = originalHeight;
                
                resolve();
            };
            img.onerror = () => {
                imageContainer.innerHTML = '<p class="placeholder">Error: Gagal memuat gambar.</p>';
                reject(new Error("Image load failed"));
            };
            img.src = imageUrl;
        });

        // Reset zoom
        setZoom(1.0);

        // Parse and display text with coordinates
        const linesData = parseXmlText(xmlDoc);
        displayTextContent(linesData);

        // Setup highlighting
        const displayedImg = imageContainer.querySelector('img');
        if (displayedImg && displayedImg.complete) {
            setupHighlighting(displayedImg, linesData);
            
            // Draw all bounding boxes if enabled
            if (showBoxes) {
                drawAllBoundingBoxes(displayedImg, linesData);
            }
        }

        // Update line count
        if (lineCountDisplay) {
            lineCountDisplay.textContent = `${linesData.length} baris`;
        }

    } catch (error) {
        console.error(`Error loading document ${docId}:`, error);
        imageContainer.innerHTML = `<p class="placeholder">Error: ${error.message}</p>`;
        textContent.innerHTML = `<p class="placeholder">Error: ${error.message}</p>`;
    }
}

function parseXmlText(xmlDoc) {
    const linesData = [];
    const textRegions = xmlDoc.getElementsByTagName('TextRegion');
    let lineNumber = 1;

    for (const region of textRegions) {
        const textLines = region.getElementsByTagName('TextLine');
        
        for (const line of textLines) {
            const coordsElement = line.getElementsByTagName('Coords')[0];
            const textEquiv = line.querySelector(':scope > TextEquiv');
            
            let lineText = "";
            let coordsPoints = "";

            if (textEquiv) {
                const unicode = textEquiv.getElementsByTagName('Unicode')[0];
                if (unicode && unicode.textContent) {
                    lineText = unicode.textContent.trim();
                }
            }
            
            if (coordsElement) {
                coordsPoints = coordsElement.getAttribute('points') || "";
            }

            if (lineText) {
                linesData.push({
                    lineNumber: lineNumber,
                    text: lineText,
                    coords: coordsPoints
                });
                lineNumber++;
            }
        }
    }

    return linesData;
}

function displayTextContent(linesData) {
    if (!textContent) return;

    textContent.innerHTML = '';

    if (linesData.length === 0) {
        textContent.innerHTML = '<p class="placeholder">Tidak ada teks ditemukan di XML.</p>';
        return;
    }

    linesData.forEach(line => {
        const span = document.createElement('span');
        span.className = 'text-line';
        span.dataset.lineNumber = line.lineNumber;
        if (line.coords) {
            span.dataset.coords = line.coords;
        }
        span.innerHTML = `<span class="line-number">${line.lineNumber}.</span>${escapeHtml(line.text)}`;
        textContent.appendChild(span);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setupHighlighting(imageElement, linesData) {
    const highlightLayer = getOrCreateHighlightLayer(imageElement);
    const originalWidth = parseFloat(imageElement.dataset.originalWidth);
    const originalHeight = parseFloat(imageElement.dataset.originalHeight);

    // Store linesData for later use
    imageElement.linesData = linesData;

    // Remove old listeners
    textContent.onmouseover = null;
    textContent.onmouseout = null;

    // Add new listeners
    textContent.addEventListener('mouseover', (event) => {
        if (event.target.classList.contains('text-line') && event.target.dataset.coords) {
            removeHighlight();
            const coordsStr = event.target.dataset.coords;
            const lineNumber = event.target.dataset.lineNumber;
            
            if (!isNaN(originalWidth) && !isNaN(originalHeight)) {
                drawHighlightShape(coordsStr, imageElement, highlightLayer, originalWidth, originalHeight, lineNumber);
                event.target.classList.add('highlight-text');
            }
        }
    });

    textContent.addEventListener('mouseout', (event) => {
        if (event.target.classList.contains('text-line')) {
            removeHighlight();
            event.target.classList.remove('highlight-text');
            
            // Redraw bounding boxes if enabled
            if (showBoxes && imageElement.linesData) {
                drawAllBoundingBoxes(imageElement, imageElement.linesData);
            }
        }
    });
}

function getOrCreateHighlightLayer(imageElement) {
    let layer = document.getElementById('highlight-layer');
    if (!layer) {
        layer = document.createElement('div');
        layer.id = 'highlight-layer';
        layer.style.position = 'absolute';
        layer.style.pointerEvents = 'none';
        layer.style.overflow = 'visible';
        imageContainer.style.position = 'relative';
        imageContainer.appendChild(layer);
    }

    // Update layer position and size
    const imgRect = imageElement.getBoundingClientRect();
    const containerRect = imageContainer.getBoundingClientRect();
    
    layer.style.top = `${imgRect.top - containerRect.top + imageContainer.scrollTop}px`;
    layer.style.left = `${imgRect.left - containerRect.left + imageContainer.scrollLeft}px`;
    layer.style.width = `${imgRect.width}px`;
    layer.style.height = `${imgRect.height}px`;
    layer.innerHTML = '';
    
    return layer;
}

function removeHighlight() {
    const layer = document.getElementById('highlight-layer');
    if (layer) {
        layer.innerHTML = '';
    }
    textContent.querySelectorAll('.highlight-text').forEach(el => {
        el.classList.remove('highlight-text');
    });
}

function drawHighlightShape(coordsStr, imageElement, layer, originalWidth, originalHeight, lineNumber) {
    if (!coordsStr || isNaN(originalWidth) || isNaN(originalHeight)) {
        return;
    }

    const pointsList = coordsStr.split(' ').map(p => {
        const xy = p.split(',');
        if (xy.length === 2) {
            return { x: parseFloat(xy[0]), y: parseFloat(xy[1]) };
        }
        return null;
    }).filter(p => p !== null);

    if (pointsList.length < 3) return;

    // Get display dimensions
    const displayWidth = parseFloat(layer.style.width);
    const displayHeight = parseFloat(layer.style.height);

    // Scale factors
    const scaleX = displayWidth / originalWidth;
    const scaleY = displayHeight / originalHeight;

    // Create SVG
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.left = '0';
    svg.style.top = '0';
    svg.style.overflow = 'visible';

    // Create polygon
    const scaledPointsStr = pointsList.map(p => `${p.x * scaleX},${p.y * scaleY}`).join(' ');
    const polygon = document.createElementNS(svgNS, "polygon");
    polygon.setAttribute("points", scaledPointsStr);
    polygon.setAttribute("class", "highlight-polygon");
    svg.appendChild(polygon);

    // Add line number label
    if (lineNumber && pointsList.length > 0) {
        const numberDiv = document.createElement('div');
        numberDiv.className = 'highlight-line-number';
        numberDiv.textContent = lineNumber;
        numberDiv.style.left = `${Math.max(0, pointsList[0].x * scaleX - 5)}px`;
        numberDiv.style.top = `${Math.max(0, pointsList[0].y * scaleY - 20)}px`;
        layer.appendChild(numberDiv);
    }

    layer.appendChild(svg);
}

function drawAllBoundingBoxes(imageElement, linesData) {
    const layer = getOrCreateHighlightLayer(imageElement);
    const originalWidth = parseFloat(imageElement.dataset.originalWidth);
    const originalHeight = parseFloat(imageElement.dataset.originalHeight);
    
    if (isNaN(originalWidth) || isNaN(originalHeight)) return;

    const displayWidth = parseFloat(layer.style.width);
    const displayHeight = parseFloat(layer.style.height);
    const scaleX = displayWidth / originalWidth;
    const scaleY = displayHeight / originalHeight;

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.left = '0';
    svg.style.top = '0';

    // Color palette for boxes
    const colors = [
        'rgba(233, 69, 96, 0.3)',   // Red
        'rgba(69, 233, 96, 0.3)',   // Green
        'rgba(69, 96, 233, 0.3)',   // Blue
        'rgba(233, 233, 69, 0.3)', // Yellow
        'rgba(233, 69, 233, 0.3)', // Magenta
        'rgba(69, 233, 233, 0.3)'  // Cyan
    ];

    linesData.forEach((line, idx) => {
        if (!line.coords) return;

        const pointsList = line.coords.split(' ').map(p => {
            const xy = p.split(',');
            return xy.length === 2 ? { x: parseFloat(xy[0]), y: parseFloat(xy[1]) } : null;
        }).filter(p => p !== null);

        if (pointsList.length < 3) return;

        const scaledPointsStr = pointsList.map(p => `${p.x * scaleX},${p.y * scaleY}`).join(' ');
        const polygon = document.createElementNS(svgNS, "polygon");
        polygon.setAttribute("points", scaledPointsStr);
        polygon.setAttribute("fill", colors[idx % colors.length]);
        polygon.setAttribute("stroke", colors[idx % colors.length].replace('0.3', '0.8'));
        polygon.setAttribute("stroke-width", "1");
        svg.appendChild(polygon);

        // Add line number
        const numDiv = document.createElement('div');
        numDiv.className = 'bbox-number';
        numDiv.textContent = line.lineNumber;
        numDiv.style.position = 'absolute';
        numDiv.style.left = `${pointsList[0].x * scaleX}px`;
        numDiv.style.top = `${Math.max(0, pointsList[0].y * scaleY - 18)}px`;
        numDiv.style.background = colors[idx % colors.length].replace('0.3', '0.9');
        numDiv.style.color = 'white';
        numDiv.style.padding = '1px 4px';
        numDiv.style.fontSize = '10px';
        numDiv.style.borderRadius = '2px';
        numDiv.style.pointerEvents = 'none';
        layer.appendChild(numDiv);
    });

    layer.appendChild(svg);
}

// Mouse wheel zoom handler
function handleMouseWheelZoom(event) {
    event.preventDefault();
    
    const img = imageContainer.querySelector('img');
    if (!img) return;
    
    // Calculate zoom direction
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    const newZoom = Math.max(0.25, Math.min(5.0, currentZoom + delta));
    
    if (newZoom !== currentZoom) {
        // Get mouse position relative to container
        const rect = imageContainer.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Calculate scroll position to keep mouse point stable
        const scrollLeft = imageContainer.scrollLeft;
        const scrollTop = imageContainer.scrollTop;
        
        // Calculate the point under cursor before zoom
        const pointX = (scrollLeft + mouseX) / currentZoom;
        const pointY = (scrollTop + mouseY) / currentZoom;
        
        // Apply new zoom
        setZoom(newZoom);
        
        // Calculate new scroll position to keep the point under cursor
        const newScrollLeft = (pointX * newZoom) - mouseX;
        const newScrollTop = (pointY * newZoom) - mouseY;
        
        imageContainer.scrollLeft = newScrollLeft;
        imageContainer.scrollTop = newScrollTop;
    }
}

// Pan/drag handlers
function handlePanStart(event) {
    // Only pan with left mouse button and not on other interactive elements
    if (event.button !== 0) return;
    
    const img = imageContainer.querySelector('img');
    if (!img) return;
    
    isPanning = true;
    panStartX = event.clientX;
    panStartY = event.clientY;
    scrollStartX = imageContainer.scrollLeft;
    scrollStartY = imageContainer.scrollTop;
    
    imageContainer.style.cursor = 'grabbing';
    event.preventDefault();
}

function handlePanMove(event) {
    if (!isPanning) return;
    
    const deltaX = event.clientX - panStartX;
    const deltaY = event.clientY - panStartY;
    
    imageContainer.scrollLeft = scrollStartX - deltaX;
    imageContainer.scrollTop = scrollStartY - deltaY;
}

function handlePanEnd() {
    if (isPanning) {
        isPanning = false;
        imageContainer.style.cursor = 'grab';
    }
}

// Zoom functions
function setZoom(level) {
    currentZoom = Math.max(0.25, Math.min(5.0, level));
    const img = imageContainer.querySelector('img');
    if (img) {
        img.style.transform = `scale(${currentZoom})`;
        img.style.transformOrigin = 'top left';
        
        // Update highlight layer after a small delay to let transform complete
        setTimeout(() => {
            if (img.linesData) {
                const layer = getOrCreateHighlightLayer(img);
                if (showBoxes) {
                    drawAllBoundingBoxes(img, img.linesData);
                }
            }
        }, 50);
    }
    if (zoomLevelDisplay) {
        zoomLevelDisplay.textContent = `${Math.round(currentZoom * 100)}%`;
    }
}

// Toggle bounding boxes
function toggleBoundingBoxes() {
    showBoxes = !showBoxes;
    toggleBoxesBtn.textContent = showBoxes ? '🔲 Hide Boxes' : '🔲 Show Boxes';
    
    const img = imageContainer.querySelector('img');
    if (img && img.linesData) {
        if (showBoxes) {
            drawAllBoundingBoxes(img, img.linesData);
        } else {
            const layer = document.getElementById('highlight-layer');
            if (layer) layer.innerHTML = '';
        }
    }
}

// Export all text
function exportAllText() {
    const lines = textContent.querySelectorAll('.text-line');
    if (lines.length === 0) {
        alert('Tidak ada teks untuk diekspor.');
        return;
    }

    let text = '';
    lines.forEach(line => {
        const num = line.querySelector('.line-number').textContent;
        const content = line.textContent.replace(num, '').trim();
        text += `${num} ${content}\n`;
    });

    // Download as file
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transkripsi.txt';
    a.click();
    URL.revokeObjectURL(url);
}

// Copy current text
function copyCurrentText() {
    const lines = textContent.querySelectorAll('.text-line');
    if (lines.length === 0) {
        alert('Tidak ada teks untuk disalin.');
        return;
    }

    let text = '';
    lines.forEach(line => {
        const content = line.textContent.replace(/^\d+\./, '').trim();
        text += content + '\n';
    });

    navigator.clipboard.writeText(text).then(() => {
        // Show feedback
        const originalText = copyTextBtn.textContent;
        copyTextBtn.textContent = '✓ Copied!';
        setTimeout(() => {
            copyTextBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Gagal menyalin teks.');
    });
}
