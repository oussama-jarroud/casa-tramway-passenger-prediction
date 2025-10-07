
    function showSpinner() {
        document.getElementById('loadingSpinner').classList.remove('hidden');
        // Optionnel: Cacher les résultats précédents si l'utilisateur soumet de nouveau
        var resultsSection = document.querySelector('.results-section');
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
        var initialChart = document.querySelector('.initial-chart');
        if (initialChart) {
            initialChart.classList.add('hidden');
        }
    }

    // Cacher le spinner si la page est chargée et qu'il est visible (en cas d'erreur par exemple)
    document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('loadingSpinner').classList.add('hidden');
    });
