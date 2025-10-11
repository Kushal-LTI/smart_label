// ==============================================================================
//           COMPLETE JAVASCRIPT FOR FULL-FEATURED CURATION PLATFORM
// ==============================================================================

document.addEventListener('DOMContentLoaded', function () {
    // --- DOM Element References ---
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('files');
    const folderInput = document.getElementById('folder');
    const patientNameInput = document.getElementById('patientName');
    const submitButton = document.getElementById('submitBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const fileLabel = document.querySelector('.file-label[for="files"] span');
    const folderLabel = document.querySelector('.file-label[for="folder"] span');
    const processingSpinner = document.getElementById('processingSpinner');
    const sessionControls = document.getElementById('sessionControls');
    const finalizeBtn = document.getElementById('finalizeBtn');
    const reportContainer = document.getElementById('reportContainer');
    const selectAllCheckbox = document.getElementById('selectAllCheckbox');
    const batchActionSelect = document.getElementById('batchActionSelect');
    const applyBatchActionBtn = document.getElementById('applyBatchActionBtn');

    let currentSessionUUID = null;
    let selectedCards = new Set();

    // --- Event Listeners ---
    fileInput.addEventListener('change', () => updateFileLabel(fileInput, fileLabel, '🖼️ Select Images'));
    folderInput.addEventListener('change', () => updateFileLabel(folderInput, folderLabel, '📁 Select Folder'));
    uploadForm.addEventListener('submit', handleUploadSubmit);
    finalizeBtn.addEventListener('click', handleFinalize);
    selectAllCheckbox.addEventListener('change', handleSelectAll);
    applyBatchActionBtn.addEventListener('click', handleBatchAction);

    // Event Delegation for dynamic content
    resultsContainer.addEventListener('submit', handleCorrectionSubmit);
    resultsContainer.addEventListener('click', function(event) {
        if (event.target.matches('.toggle-explanation-btn')) {
            handleExplanationToggle(event);
        }
        if (event.target.matches('.card-checkbox')) {
            handleCardSelection(event);
        }
    });

    // --- Handler Functions ---

    function updateFileLabel(input, labelElement, defaultText) {
        if (input.files.length > 0) {
            labelElement.textContent = `${input.files.length} file(s) selected`;
            labelElement.parentElement.classList.add('selected');
        } else {
            labelElement.textContent = defaultText;
            labelElement.parentElement.classList.remove('selected');
        }
    }

    function handleUploadSubmit(event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);

        // Reset UI
        resultsContainer.innerHTML = '';
        reportContainer.innerHTML = '';
        sessionControls.style.display = 'none';
        processingSpinner.style.display = 'flex';
        submitButton.disabled = true;

        fetch('/upload', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                currentSessionUUID = data.session_uuid;
                data.results.forEach(result => {
                    resultsContainer.insertAdjacentHTML('beforeend', createCardHTML(result));
                });
                if (data.results.length > 0) {
                    sessionControls.style.display = 'flex';
                }
            })
            .catch(error => console.error('Upload Error:', error))
            .finally(() => {
                processingSpinner.style.display = 'none';
                submitButton.disabled = false;
            });
    }

    function handleCorrectionSubmit(event) {
        // Target the form that was submitted
        if (event.target && event.target.matches('.hitl-form')) {
            // CRUCIAL: Prevent the default form submission (page reload)
            event.preventDefault(); 
            
            const form = event.target;
            const card = form.closest('.card');
            
            const correctLabel = form.querySelector('.hitl-select').value;
            const tempFilename = form.querySelector('input[name="temp_filename"]').value;
            const submitBtn = form.querySelector('.hitl-submit-btn');
            
            submitBtn.textContent = 'Saving...';
            submitBtn.disabled = true;

            fetch('/correct_label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_uuid: currentSessionUUID,
                    temp_filename: tempFilename,
                    new_label: correctLabel
                })
            })
            .then(response => response.json())
            .then(responseData => {
                if (responseData.success) {
                    card.classList.remove('hitl-card');
                    card.classList.add('card-saved');
                    // We only need to update the button and select state, not the whole card
                    const select = form.querySelector('.hitl-select');
                    select.disabled = true;
                    submitBtn.textContent = 'Confirmed';
                    submitBtn.classList.add('confirmed');
                } else {
                    alert('Error: ' + responseData.message);
                    submitBtn.textContent = 'Confirm Label';
                    submitBtn.disabled = false;
                }
            });
        }
    }

    function handleCardSelection(event) {
        const checkbox = event.target;
        const card = checkbox.closest('.card');
        const tempFilename = card.dataset.tempFilename;

        if (checkbox.checked) {
            selectedCards.add(tempFilename);
            card.classList.add('card-selected');
        } else {
            selectedCards.delete(tempFilename);
            card.classList.remove('card-selected');
        }
    }

    function handleSelectAll() {
        const allCheckboxes = resultsContainer.querySelectorAll('.card-checkbox');
        const allCards = resultsContainer.querySelectorAll('.card');

        if (selectAllCheckbox.checked) {
            allCheckboxes.forEach(cb => cb.checked = true);
            allCards.forEach(card => {
                selectedCards.add(card.dataset.tempFilename);
                card.classList.add('card-selected');
            });
        } else {
            allCheckboxes.forEach(cb => cb.checked = false);
            allCards.forEach(card => card.classList.remove('card-selected'));
            selectedCards.clear();
        }
    }

    function handleBatchAction() {
        const action = batchActionSelect.value;
        if (!action || selectedCards.size === 0) {
            alert("Please select an action and at least one image.");
            return;
        }

        selectedCards.forEach(tempFilename => {
            const card = resultsContainer.querySelector(`.card[data-temp-filename="${tempFilename}"]`);
            if (card) {
                const select = card.querySelector('.hitl-select');
                if (select) {
                    select.value = action;
                    // Trigger a change event to notify the backend
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });

        // Optional: clear selection after applying
        selectAllCheckbox.checked = false;
        handleSelectAll();
    }

    function handleFinalize() {
        finalizeBtn.textContent = 'Generating Report...';
        finalizeBtn.disabled = true;

        fetch('/finalize_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_uuid: currentSessionUUID })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Clear the review cards and show the report
                    resultsContainer.innerHTML = '';
                    sessionControls.style.display = 'none';
                    // The new displayReport now takes the report_id
                    displayReport(data);
                } else {
                    alert('Error finalizing session: ' + data.message);
                }
            })
            .finally(() => {
                finalizeBtn.textContent = 'Finish & Save Report';
                finalizeBtn.disabled = false;
            });
    }

    function handleExplanationToggle(event) {
        const button = event.target;
        const card = button.closest('.card');
        const explanationContainer = card.querySelector('.explanation-container');

        // If container is already open, just close it
        if (explanationContainer.classList.contains('visible')) {
            explanationContainer.classList.remove('visible');
            button.textContent = 'View AI Explanation';
            return;
        }

        // If it has already been loaded, just re-open it
        if (explanationContainer.dataset.loaded === 'true') {
            explanationContainer.classList.add('visible');
            button.textContent = 'Hide Explanation';
            return;
        }

        // Otherwise, fetch the explanation from the server
        const tempFilename = card.dataset.tempFilename;
        button.disabled = true;
        button.textContent = 'Loading...';

        fetch('/get_explanation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temp_filename: tempFilename })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const result = JSON.parse(card.dataset.result); // Get stored result data
                    explanationContainer.innerHTML = createExplanationViewHTML(result, data.grad_cam_urls);
                    explanationContainer.classList.add('visible');
                    explanationContainer.dataset.loaded = 'true';
                    button.textContent = 'Hide Explanation';
                } else {
                    alert('Error: ' + data.message);
                    button.textContent = 'View AI Explanation';
                }
            })
            .catch(error => console.error('Explanation Error:', error))
            .finally(() => {
                button.disabled = false;
            });
    }


    function displayReport(data) {
        let summaryRows = Object.entries(data.summary).map(([label, count]) => `
        <tr><td>${label.charAt(0).toUpperCase() + label.slice(1)}</td><td>${count}</td></tr>
    `).join('');

        const totalCount = Object.values(data.summary).reduce((sum, count) => sum + count, 0);

        reportContainer.innerHTML = `
        <div class="report-card">
            <h2>Curation Report Generated</h2>
            <h3>Patient: ${data.patient_name}</h3>
            <div class="report-summary">
                <table>
                    <thead><tr><th>Cell Type</th><th>Count</th></tr></thead>
                    <tbody>
                        ${summaryRows}
                        <tr class="total-row"><td><strong>Total Cells</strong></td><td><strong>${totalCount}</strong></td></tr>
                    </tbody>
                </table>
                <!-- Use the new report_id for the download link -->
                <a href="/download_report/${data.report_id}" class="download-btn">Download PDF Report</a>
            </div>
        </div>
    `;
    }

    function createCardHTML(result) {
        const cardClass = result.hitl_required ? 'hitl-card' : 'auto-labeled-card';
        const resultDataString = JSON.stringify(result).replace(/"/g, '&quot;');

        let predictionDetailsHTML = result.individual_preds.map(p => `
        <li><strong>${p.model_name}:</strong> ${p.predicted_class} <em>(${(p.confidence).toFixed(1)}%)</em></li>
    `).join('');

        const optionsHTML = config.CLASS_LABELS.map(className =>
            `<option value="${className}" ${className === result.final_prediction ? 'selected' : ''}>
            ${className.charAt(0).toUpperCase() + className.slice(1)}
        </option>`
        ).join('');

        // Determine which button to show
        const actionButtonHTML = result.hitl_required ?
            `<button type="submit" class="hitl-submit-btn">Confirm Label</button>` :
            `<button type="button" class="confirm-btn" disabled>Auto-Confirmed</button>`;

        return `
        <div class="card ${cardClass}" data-temp-filename="${result.temp_filename}" data-result="${resultDataString}">
            <div class="card-header">
                <input type="checkbox" class="card-checkbox" title="Select this card">
                <div class="reason-chip">${result.reason}</div>
            </div>
            <img src="${result.temp_image_path}" class="main-image" alt="Cell Image">
            <div class="card-content">
                 <div class="prediction-details">
                    <h4>Ensemble Prediction</h4>
                    <ul>
                        ${predictionDetailsHTML}
                        <li class="summary"><strong>Avg. Confidence:</strong> ${result.avg_confidence.toFixed(1)}%</li>
                    </ul>
                </div>
                <form class="hitl-form">
                    <input type="hidden" name="temp_filename" value="${result.temp_filename}">
                    <label class="hitl-label">Final Label:</label>
                    <div class="form-row">
                        <select class="hitl-select" ${!result.hitl_required ? 'disabled' : ''}>${optionsHTML}</select>
                        ${actionButtonHTML}
                    </div>
                </form>
            </div>
            <div class="card-footer">
                <button class="toggle-explanation-btn">View AI Explanation</button>
            </div>
            <div class="explanation-container">
                <!-- Grad-CAM content will be loaded here -->
            </div>
        </div>
    `;
    }

    function createExplanationViewHTML(result, gradCamUrls) {
        const [pred1, pred2] = result.individual_preds;
        return `
        <div class="hitl-comparison-grid">
            <div class="model-view">
                <img src="${gradCamUrls[pred1.model_name]}" class="cam-image">
                <div class="model-prediction"><strong>${pred1.model_name}</strong> predicts: <span>${pred1.predicted_class}</span></div>
            </div>
            <div class="model-view">
                <img src="${gradCamUrls[pred2.model_name]}" class="cam-image">
                <div class="model-prediction"><strong>${pred2.model_name}</strong> predicts: <span>${pred2.predicted_class}</span></div>
            </div>
        </div>
    `;
    }
});