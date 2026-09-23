// Coder UI Interactive Enhancements
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");

    if (form) {
        form.addEventListener("submit", () => {
            // Add terminal processing visual feedback
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>COMPUTING INFERENCE...`;
        });
    }
});