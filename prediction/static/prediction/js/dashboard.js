function setupDashboardInteractions() {
    const form = document.getElementById('stockForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');

    form?.addEventListener('submit', () => {
        if (btnText) btnText.innerText = 'İşleniyor';
        btnSpinner?.classList.remove('d-none');
        if (analyzeBtn) analyzeBtn.disabled = true;
    });

    document.querySelectorAll('[data-chart-toggle]').forEach(toggle => {
        toggle.addEventListener('change', event => {
            const datasetKey = event.target.dataset.chartToggle;
            const chart = window.stockPriceChart;
            if (!chart) return;
            const dataset = chart.data.datasets.find(item => item.datasetKey === datasetKey);
            if (!dataset) return;
            dataset.hidden = !event.target.checked;
            chart.update();
        });
    });
}

document.addEventListener('DOMContentLoaded', setupDashboardInteractions);
