/* ===================================================================
   StockVision AI — Frontend Logic
   All 5 charts: Actual vs Predicted, Residual, Rolling Dir Accuracy,
   Prediction Error Density, Actual vs Predicted Direction
   =================================================================== */

let charts = {};

// ---- Chart.js global config for dark theme ----
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;

// ---- Boot ----
document.addEventListener('DOMContentLoaded', loadStocks);

let allStockGroups = {};   // { dataset: [stock, …] }
let selectedStock = null;

async function loadStocks() {
    const res = await fetch('/api/stocks');
    const stocks = await res.json();

    // Group by dataset
    allStockGroups = {};
    stocks.forEach(s => {
        if (!allStockGroups[s.Dataset]) allStockGroups[s.Dataset] = [];
        allStockGroups[s.Dataset].push(s.Stock);
    });

    buildDropdownList('');

    // Select first stock
    const firstDataset = Object.keys(allStockGroups)[0];
    if (firstDataset && allStockGroups[firstDataset].length) {
        selectStock(allStockGroups[firstDataset][0]);
    }

    // --- Event listeners ---
    const trigger = document.getElementById('selectTrigger');
    const dropdown = document.getElementById('selectDropdown');
    const searchInput = document.getElementById('stockSearch');
    const wrapper = document.getElementById('customSelect');

    trigger.addEventListener('click', () => {
        wrapper.classList.toggle('open');
        if (wrapper.classList.contains('open')) {
            searchInput.value = '';
            buildDropdownList('');
            setTimeout(() => searchInput.focus(), 50);
        }
    });

    searchInput.addEventListener('input', () => {
        buildDropdownList(searchInput.value.trim());
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
        if (!wrapper.contains(e.target)) {
            wrapper.classList.remove('open');
        }
    });

    // Keyboard: Escape to close
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            wrapper.classList.remove('open');
        }
    });
}

function buildDropdownList(filter) {
    const list = document.getElementById('stockList');
    const lowerFilter = filter.toLowerCase();
    let html = '';
    let hasResults = false;

    Object.entries(allStockGroups).forEach(([dataset, stocks]) => {
        const matched = stocks.filter(s => s.toLowerCase().includes(lowerFilter));
        if (matched.length === 0) return;
        hasResults = true;

        html += `<li class="custom-select__group-label">${dataset}</li>`;
        matched.forEach(stock => {
            const isActive = stock === selectedStock ? ' active' : '';
            html += `<li class="custom-select__option${isActive}" data-value="${stock}">${stock}</li>`;
        });
    });

    if (!hasResults) {
        html = `<li class="custom-select__no-results">No stocks found</li>`;
    }

    list.innerHTML = html;

    // Attach click handlers
    list.querySelectorAll('.custom-select__option').forEach(el => {
        el.addEventListener('click', () => {
            selectStock(el.dataset.value);
            document.getElementById('customSelect').classList.remove('open');
        });
    });
}

function selectStock(stock) {
    selectedStock = stock;
    document.getElementById('selectText').textContent = stock;

    // Update active state visually
    document.querySelectorAll('.custom-select__option').forEach(el => {
        el.classList.toggle('active', el.dataset.value === stock);
    });

    loadPrediction(stock);
}

async function loadPrediction(stock) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('active');

    try {
        const res = await fetch(`/api/predict/${encodeURIComponent(stock)}`);
        const data = await res.json();

        if (data.error) {
            document.getElementById('summary').innerHTML =
                `<div class="cards"><div class="card" style="grid-column:1/-1;text-align:center;padding:32px;">
                <p style="color:var(--accent-rose);font-weight:600;">${data.error}</p></div></div>`;
            overlay.classList.remove('active');
            return;
        }

        renderSummary(data);
        renderAllCharts(data);
    } catch (err) {
        console.error(err);
    } finally {
        overlay.classList.remove('active');
    }
}

// ===== SUMMARY CARDS =====
function renderSummary(data) {
    const changePct = data.predicted_change_pct;
    const changeDir = changePct >= 0 ? 'positive' : 'negative';
    const changeArrow = changePct >= 0 ? '▲' : '▼';
    const changeSign = changePct >= 0 ? '+' : '';

    const volStr = data.volume != null
        ? (data.volume >= 1e6 ? (data.volume / 1e6).toFixed(2) + 'M'
            : data.volume >= 1e3 ? (data.volume / 1e3).toFixed(1) + 'K'
                : data.volume.toLocaleString())
        : '—';

    const modelName = data.model_name || 'Unknown';
    const modelColor = modelName === 'iTransformer' ? '#22d3ee' : '#a78bfa';

    const liveWarning = data.is_live === false ? 
        `<div style="grid-column: 1 / -1; background: rgba(251, 113, 133, 0.08); border: 1px solid rgba(251, 113, 133, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 4px; color: #fb7185; display: flex; align-items: center; gap: 8px; font-weight: 500; font-size: 0.85rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            Live market data unavailable. Forecasting using offline historical CSV data.
        </div>` : '';

    document.getElementById('summary').innerHTML = `
        <div class="cards">
            ${liveWarning}
            <div class="card card--date">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                    <line x1="16" y1="2" x2="16" y2="6"/>
                    <line x1="8" y1="2" x2="8" y2="6"/>
                    <line x1="3" y1="10" x2="21" y2="10"/>
                </svg>
                <span class="card-label">Prediction Date</span>
                <span class="card-value">${data.date}</span>
                <span style="margin-left:auto;font-size:0.78rem;color:var(--text-muted);">${data.dataset}</span>
            </div>

            <div class="card">
                <div class="card-label">Stock</div>
                <div class="card-value">${data.stock}</div>
            </div>
            <div class="card">
                <div class="card-label">Open</div>
                <div class="card-value">${data.open.toFixed(2)}</div>
            </div>
            <div class="card">
                <div class="card-label">High</div>
                <div class="card-value">${data.high.toFixed(2)}</div>
            </div>
            <div class="card">
                <div class="card-label">Low</div>
                <div class="card-value">${data.low.toFixed(2)}</div>
            </div>
            <div class="card">
                <div class="card-label">Close</div>
                <div class="card-value">${data.close.toFixed(2)}</div>
            </div>
            <div class="card">
                <div class="card-label">Volume</div>
                <div class="card-value">${volStr}</div>
            </div>
            <div class="card card--highlight">
                <div class="card-label">Predicted Price</div>
                <div class="card-value">${data.predicted_price.toFixed(2)}</div>
            </div>
            <div class="card card--highlight">
                <div class="card-label">Expected Change</div>
                <div class="card-value ${changeDir}">${changeArrow} ${changeSign}${changePct.toFixed(2)}%</div>
            </div>
            <div class="card card--model">
                <div class="card-label">Active Model</div>
                <div class="card-value" style="color:${modelColor};font-weight:700;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px;">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                    </svg>
                    ${modelName}
                </div>
            </div>
        </div>
    `;
}

// ===== RENDER ALL CHARTS =====

function renderAllCharts(data) {
    const labels = data.history.map(x => x.Date);
    const actual = data.history.map(x => x.Close);
    const predicted = data.history.map(x => x.Predicted);
    const residual = data.history.map(x => x.Residual);
    const rollingAcc = data.history.map(x => x.RollingDirAcc);
    const actualDir = data.history.map(x => x.ActualDir);
    const predDir = data.history.map(x => x.PredDir);
    const priceMovement = data.history.map(x => x.PriceMovement);
    const predMovement = data.history.map(x => x.PredMovement);
    const modelName = data.model_name || 'Model';

    drawActualVsPredicted(labels, actual, predicted);
    drawResidual(labels, residual, priceMovement, predMovement, modelName);
    drawRollingAccuracy(labels, rollingAcc);
    drawDensity(residual);
    drawDirectionComparison(labels, actualDir, predDir);
}

// ===== 1. ACTUAL VS PREDICTED PRICE =====
function drawActualVsPredicted(labels, actual, predicted) {
    if (charts.actual) charts.actual.destroy();

    charts.actual = new Chart(document.getElementById('actualChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Actual',
                    data: actual,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#6366f1',
                    tension: 0.3,
                    fill: true,
                },
                {
                    label: 'Predicted',
                    data: predicted,
                    borderColor: '#22d3ee',
                    backgroundColor: 'rgba(34, 211, 238, 0.05)',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#22d3ee',
                    tension: 0.3,
                    fill: false,
                }
            ]
        },
        options: chartOptions('Price')
    });
}

// ===== 2. RESIDUAL ERROR + PRICE MOVEMENT (notebook 7D + 7E) =====
function drawResidual(labels, residual, priceMovement, predMovement, modelName) {
    if (charts.residual) charts.residual.destroy();

    const datasets = [
        {
            label: 'Residual Error',
            data: residual,
            backgroundColor: residual.map(v => v >= 0 ? 'rgba(52, 211, 153, 0.45)' : 'rgba(251, 113, 133, 0.45)'),
            borderColor: residual.map(v => v >= 0 ? '#34d399' : '#fb7185'),
            borderWidth: 1,
            borderRadius: 3,
            type: 'bar',
            order: 2,
        },
    ];

    // Overlay price movement lines if available (notebook 7E)
    if (priceMovement && predMovement) {
        datasets.push(
            {
                label: 'Actual Movement',
                data: priceMovement,
                borderColor: '#f1f5f9',
                borderWidth: 1.8,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.35,
                type: 'line',
                order: 1,
                yAxisID: 'y',
            },
            {
                label: `${modelName} Pred Movement`,
                data: predMovement,
                borderColor: '#22d3ee',
                borderWidth: 1.8,
                borderDash: [5, 3],
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.35,
                type: 'line',
                order: 0,
                yAxisID: 'y',
            },
        );
    }

    charts.residual = new Chart(document.getElementById('residualChart'), {
        type: 'bar',
        data: { labels, datasets },
        options: {
            ...chartOptions('Error / Movement'),
            plugins: {
                ...chartOptions('Error / Movement').plugins,
                annotation: undefined,
            }
        }
    });
}

// ===== 3. ROLLING DIRECTION ACCURACY =====
function drawRollingAccuracy(labels, rollingAcc) {
    if (charts.rollingAcc) charts.rollingAcc.destroy();

    const gradientCanvas = document.getElementById('rollingAccChart');
    const ctx = gradientCanvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(52, 211, 153, 0.25)');
    gradient.addColorStop(1, 'rgba(52, 211, 153, 0)');

    charts.rollingAcc = new Chart(gradientCanvas, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Direction Accuracy (%)',
                    data: rollingAcc,
                    borderColor: '#34d399',
                    backgroundColor: gradient,
                    borderWidth: 2.5,
                    borderDash: [6, 3],
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#34d399',
                    tension: 0.4,
                    fill: true,
                },
                {
                    label: '50% Baseline',
                    data: Array(labels.length).fill(50),
                    borderColor: 'rgba(251, 191, 36, 0.4)',
                    borderWidth: 1.5,
                    borderDash: [8, 4],
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            ...chartOptions('Accuracy (%)'),
            scales: {
                ...chartOptions('Accuracy (%)').scales,
                y: {
                    ...chartOptions('Accuracy (%)').scales.y,
                    min: 0,
                    max: 100,
                }
            }
        }
    });
}

// ===== 4. PREDICTION ERROR DENSITY =====
function drawDensity(residual) {
    if (charts.density) charts.density.destroy();

    const clean = residual.filter(v => Number.isFinite(v));
    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const bins = 30;
    const step = (max - min) / bins || 1;

    const counts = Array(bins).fill(0);
    clean.forEach(v => {
        const idx = Math.min(bins - 1, Math.floor((v - min) / step));
        counts[idx]++;
    });

    const binLabels = counts.map((_, i) => (min + i * step).toFixed(2));

    const ctx = document.getElementById('densityChart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.7)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.2)');

    charts.density = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: 'Frequency',
                data: counts,
                backgroundColor: gradient,
                borderColor: 'rgba(139, 92, 246, 0.6)',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            ...chartOptions('Count'),
            scales: {
                ...chartOptions('Count').scales,
                x: {
                    ...chartOptions('Count').scales.x,
                    ticks: {
                        ...chartOptions('Count').scales.x.ticks,
                        maxTicksLimit: 10,
                    }
                }
            }
        }
    });
}

// ===== 5. ACTUAL VS PREDICTED DIRECTION =====
function drawDirectionComparison(labels, actualDir, predDir) {
    if (charts.direction) charts.direction.destroy();

    // For each point: match=green, mismatch=red
    const matchColors = actualDir.map((a, i) => a === predDir[i] ? 'rgba(52, 211, 153, 0.7)' : 'rgba(251, 113, 133, 0.7)');

    charts.direction = new Chart(document.getElementById('directionChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Actual Direction',
                    data: actualDir.map(d => d === 1 ? 1 : -1),
                    backgroundColor: actualDir.map(d => d === 1 ? 'rgba(99, 102, 241, 0.7)' : 'rgba(99, 102, 241, 0.3)'),
                    borderColor: '#6366f1',
                    borderWidth: 0,
                    borderRadius: 3,
                    barPercentage: 0.6,
                    categoryPercentage: 0.8,
                },
                {
                    label: 'Predicted Direction',
                    data: predDir.map(d => d === 1 ? 1 : -1),
                    backgroundColor: predDir.map((d, i) => {
                        const match = d === actualDir[i];
                        return match ? 'rgba(52, 211, 153, 0.7)' : 'rgba(251, 113, 133, 0.7)';
                    }),
                    borderColor: predDir.map((d, i) => d === actualDir[i] ? '#34d399' : '#fb7185'),
                    borderWidth: 0,
                    borderRadius: 3,
                    barPercentage: 0.6,
                    categoryPercentage: 0.8,
                }
            ]
        },
        options: {
            ...chartOptions('Direction'),
            scales: {
                ...chartOptions('Direction').scales,
                y: {
                    ...chartOptions('Direction').scales.y,
                    min: -1.5,
                    max: 1.5,
                    ticks: {
                        color: '#64748b',
                        callback: v => v === 1 ? '▲ Up' : v === -1 ? '▼ Down' : '',
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.06)',
                    }
                }
            },
            plugins: {
                ...chartOptions('Direction').plugins,
                tooltip: {
                    ...chartOptions('Direction').plugins.tooltip,
                    callbacks: {
                        label: ctx => {
                            const dir = ctx.raw > 0 ? 'Up ▲' : 'Down ▼';
                            return `${ctx.dataset.label}: ${dir}`;
                        }
                    }
                }
            }
        }
    });
}

// ===== SHARED CHART OPTIONS =====
function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#94a3b8',
                    font: { size: 11, weight: 500 },
                }
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                titleColor: '#f1f5f9',
                bodyColor: '#cbd5e1',
                borderColor: 'rgba(99, 102, 241, 0.3)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { weight: 600 },
                bodyFont: { size: 12 },
            }
        },
        scales: {
            x: {
                ticks: {
                    color: '#64748b',
                    maxRotation: 45,
                    maxTicksLimit: 15,
                    font: { size: 10 },
                },
                grid: { display: false }
            },
            y: {
                ticks: { color: '#64748b', font: { size: 10 } },
                title: {
                    display: true,
                    text: yLabel,
                    color: '#64748b',
                    font: { size: 11, weight: 500 },
                },
                grid: { color: 'rgba(99, 102, 241, 0.06)' }
            }
        }
    };
}
