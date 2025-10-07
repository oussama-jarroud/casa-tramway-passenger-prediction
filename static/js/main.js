document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    if (form && loading) {
        form.addEventListener('submit', function() {
            loading.style.display = 'block';
        });
    }
});
