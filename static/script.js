let selectedFile = null;
let currentStep = 0;
let isValidating = false;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewCard = document.getElementById('previewCard');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const validateBtn = document.getElementById('validateBtn');
const loadingCard = document.getElementById('loadingCard');
const resultsCard = document.getElementById('resultsCard');
const instructionsCard = document.getElementById('instructionsCard');
const toast = document.getElementById('toast');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    cycleFeaturesOnLanding();
}

function setupEventListeners() {
    // Only set up file-related listeners if we're on the validator page
    if (uploadArea && fileInput) {
        // File input change
        fileInput.addEventListener('change', handleFileSelect);
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Validate button
        if (validateBtn) {
            validateBtn.addEventListener('click', startValidation);
        }
    }
    
    // Prevent default drag behaviors on the entire document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// Feature cycling for landing page
function cycleFeaturesOnLanding() {
    const featureCards = document.querySelectorAll('.feature-card');
    if (featureCards.length === 0) return;
    
    let currentFeature = 0;
    
    setInterval(() => {
        featureCards.forEach(card => card.classList.remove('active-feature'));
        featureCards[currentFeature].classList.add('active-feature');
        currentFeature = (currentFeature + 1) % featureCards.length;
    }, 3000);
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file (JPG, PNG, etc.)', 'error');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast('File size should be less than 10MB', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        if (previewImage) {
            previewImage.src = e.target.result;
        }
        if (fileName) {
            fileName.textContent = file.name;
        }
        if (fileSize) {
            fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(1)}MB`;
        }
        
        showPreview();
        hideResults();
        showToast('File uploaded successfully!', 'success');
    };
    reader.readAsDataURL(file);
}

// UI state management
function showPreview() {
    if (previewCard) {
        previewCard.style.display = 'block';
        previewCard.classList.add('animate-slide-in-up');
    }
}

function hidePreview() {
    if (previewCard) {
        previewCard.style.display = 'none';
    }
}

function showLoading() {
    if (loadingCard) {
        loadingCard.style.display = 'block';
        loadingCard.classList.add('animate-slide-in-right');
    }
    if (instructionsCard) {
        instructionsCard.style.display = 'none';
    }
    hideResults();
}

function hideLoading() {
    if (loadingCard) {
        loadingCard.style.display = 'none';
    }
}

function showResults() {
    if (resultsCard) {
        resultsCard.style.display = 'block';
        resultsCard.classList.add('animate-slide-in-up');
    }
    if (instructionsCard) {
        instructionsCard.style.display = 'none';
    }
    hideLoading();
}

function hideResults() {
    if (resultsCard) {
        resultsCard.style.display = 'none';
    }
    if (instructionsCard) {
        instructionsCard.style.display = 'block';
    }
}

// Validation process
async function startValidation() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }
    
    if (isValidating) return;
    
    isValidating = true;
    currentStep = 0;
    
    // Update button state
    if (validateBtn) {
        validateBtn.disabled = true;
        validateBtn.innerHTML = '<div class="spinner"></div> Analyzing...';
    }
    
    showLoading();
    
    try {
        // Simulate processing steps
        await simulateProcessingSteps();
        
        // Call your API here
        const result = await validateWithAPI(selectedFile);
        
        // Show results
        displayResults(result);
        showResults();
        
    } catch (error) {
        console.error('Validation error:', error);
        let errorMessage = 'An error occurred during validation. Please try again.';
        
        // Provide more specific error messages
        if (error.message.includes('Validation failed:')) {
            errorMessage = error.message.replace('Validation failed: ', '');
        } else if (error.message.includes('HTTP error! status: 500')) {
            errorMessage = 'Server error. Please try again later.';
        } else if (error.message.includes('HTTP error! status: 400')) {
            errorMessage = 'Invalid image format. Please try a different image.';
        } else if (error.message.includes('fetch')) {
            errorMessage = 'Network error. Please check your connection and try again.';
        }
        
        showToast(errorMessage, 'error');
    } finally {
        isValidating = false;
        if (validateBtn) {
            validateBtn.disabled = false;
            validateBtn.innerHTML = '<i class="fas fa-search"></i> Validate ID Card';
        }
    }
}

async function simulateProcessingSteps() {
    const steps = document.querySelectorAll('.step-item');
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    
    for (let i = 0; i < steps.length; i++) {
        // Update step status
        steps.forEach((step, index) => {
            if (index <= i) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
        
        // Update progress
        const progress = ((i + 1) / steps.length) * 100;
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
        
        // Wait for step completion
        await new Promise(resolve => setTimeout(resolve, 800));
    }
}

// API Integration - Connect to FastAPI backend
async function validateWithAPI(file) {
    try {
        // Convert file to base64
        const base64 = await fileToBase64(file);
        
        // Prepare request data according to ValidateIDRequest schema
        const requestData = {
            user_id: 'web_user_' + Date.now(),
            image_base64: base64.split(',')[1] // Remove data:image/...;base64, prefix
        };
        
        // Call the FastAPI backend endpoint
        const response = await fetch('/validate-id', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Transform the response to match frontend expectations
        return {
            label: result.label,
            status: result.status,
            validation_score: result.validation_score,
            ocr_fields_detected: result.ocr_fields_detected,
            ocr_confidence: result.ocr_confidence,
            reason: result.reason,
            threshold: result.threshold,
            ocr_text_sample: result.ocr_text_sample
        };
        
    } catch (error) {
        console.error('API call failed:', error);
        throw new Error(`Validation failed: ${error.message}`);
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

function displayResults(data) {
    // Update result status icon and badge
    const resultStatus = document.getElementById('resultStatus');
    const resultBadge = document.getElementById('resultBadge');
    const resultMessage = document.getElementById('resultMessage');
    
    if (resultStatus) {
        resultStatus.className = `result-status ${data.label}`;
        const iconMap = {
            'genuine': 'fas fa-check-circle',
            'suspicious': 'fas fa-exclamation-triangle',
            'fake': 'fas fa-times-circle'
        };
        resultStatus.innerHTML = `<i class="${iconMap[data.label] || 'fas fa-question-circle'}"></i>`;
    }
    
    if (resultBadge) {
        resultBadge.textContent = data.label.charAt(0).toUpperCase() + data.label.slice(1);
        resultBadge.className = `result-badge ${data.label}`;
    }
    
    if (resultMessage) {
        resultMessage.textContent = data.status;
    }
    
    // Update metrics
    const confidenceScore = document.getElementById('confidenceScore');
    const confidenceFill = document.getElementById('confidenceFill');
    const ocrFields = document.getElementById('ocrFields');
    const ocrConfidence = document.getElementById('ocrConfidence');
    
    if (confidenceScore) {
        confidenceScore.textContent = `${(data.validation_score * 100).toFixed(1)}%`;
    }
    
    if (confidenceFill) {
        confidenceFill.style.width = `${data.validation_score * 100}%`;
    }
    
    if (ocrFields) {
        ocrFields.textContent = data.ocr_fields_detected ? `${data.ocr_fields_detected}/4` : 'N/A';
    }
    
    if (ocrConfidence) {
        ocrConfidence.textContent = data.ocr_confidence ? 
            `${(data.ocr_confidence * 100).toFixed(1)}% confidence` : 'Not available';
    }
    
    // Update summary
    const summaryText = document.getElementById('summaryText');
    if (summaryText) {
        let summary = data.reason || 'No specific reason provided';
        
        // Add OCR text sample if available
        if (data.ocr_text_sample && data.ocr_text_sample.trim()) {
            summary += `\n\nExtracted text: "${data.ocr_text_sample.substring(0, 100)}${data.ocr_text_sample.length > 100 ? '...' : ''}"`;
        }
        
        summaryText.textContent = summary;
    }
    
    // Log detailed results for debugging
    console.log('Validation results:', data);
}

// Reset validator to initial state
function resetValidator() {
    selectedFile = null;
    currentStep = 0;
    isValidating = false;
    
    // Reset file input
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Hide cards
    hidePreview();
    hideLoading();
    hideResults();
    
    // Show instructions
    if (instructionsCard) {
        instructionsCard.style.display = 'block';
    }
    
    // Reset button
    if (validateBtn) {
        validateBtn.disabled = false;
        validateBtn.innerHTML = '<i class="fas fa-search"></i> Validate ID Card';
    }
    
    showToast('Ready for new validation', 'success');
}

// Toast notification system
function showToast(message, type = 'success') {
    if (!toast) return;
    
    const toastIcon = document.querySelector('.toast-icon');
    const toastMessage = document.querySelector('.toast-message');
    
    if (toastIcon) {
        toastIcon.className = type === 'error' ? 
            'toast-icon fas fa-exclamation-circle' : 
            'toast-icon fas fa-check-circle';
        toastIcon.style.color = type === 'error' ? 'var(--error)' : 'var(--success)';
    }
    
    if (toastMessage) {
        toastMessage.textContent = message;
    }
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Utility functions for animations and effects
function addLoadingSpinner() {
    const style = document.createElement('style');
    style.textContent = `
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--primary);
            border-top: 2px solid transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 8px;
        }
    `;
    document.head.appendChild(style);
}

// Initialize spinner styles
addLoadingSpinner();

// Export functions for use in HTML
window.resetValidator = resetValidator;
window.startValidation = startValidation;

// Handle escape key to reset
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        resetValidator();
    }
});

// Handle window resize for mobile optimization
window.addEventListener('resize', function() {
    // Add any responsive adjustments here if needed
});

console.log('AI ID Card Validator initialized successfully!');
console.log('To integrate your API, modify the validateWithAPI function in script.js');