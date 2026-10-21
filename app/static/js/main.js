/**
 * ============================================================================
 * CLASSIC & CLEAN ML CAPESTONE CLIENT LOGIC
 * ============================================================================
 */

let appState = {
    currentDataset: null,
    presets: {},
    benchmarks: null,
    currentResults: null,
    activeChartTab: 'multimetric',
    currentWorkspaceMode: 'inference',
    chartInstance: null
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const pRes = await fetch('/api/presets');
        if (pRes.ok) {
            appState.presets = await pRes.json();
        }
    } catch (err) {
        console.error('Failed to load presets:', err);
    }

    try {
        const bRes = await fetch('/api/benchmarks');
        if (bRes.ok) {
            appState.benchmarks = await bRes.json();
        }
    } catch (err) {
        console.error('Failed to load benchmarks:', err);
    }
});

function switchWorkspaceMode(mode) {
    appState.currentWorkspaceMode = mode;
}

/**
 * Switch from View 1 (Dataset Selection) to View 2 (Input & Comparative Workspace)
 */
function selectDataset(track) {
    appState.currentDataset = track;
    appState.currentResults = null;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    const selectionView = document.getElementById('datasetSelectionView');
    const workspaceView = document.getElementById('workspaceView');
    const resultsSection = document.getElementById('resultsSection');

    if (selectionView) selectionView.style.display = 'none';
    if (workspaceView) workspaceView.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';

    // Toggle specific form
    const formReg = document.getElementById('formRegression');
    const formClf = document.getElementById('formClassification');
    const formClust = document.getElementById('formClustering');

    if (formReg) formReg.style.display = (track === 'regression') ? 'block' : 'none';
    if (formClf) formClf.style.display = (track === 'classification') ? 'block' : 'none';
    if (formClust) formClust.style.display = (track === 'clustering') ? 'block' : 'none';

    // Update track tabs
    ['regression', 'classification', 'clustering'].forEach(t => {
        const tabBtn = document.getElementById(`trackTab_${t}`);
        if (tabBtn) {
            if (t === track) {
                tabBtn.classList.add('active');
            } else {
                tabBtn.classList.remove('active');
            }
        }
    });

    // Update active dataset headers & pills
    const pill = document.getElementById('activeDatasetPill');
    const title = document.getElementById('activeDatasetTitle');
    const desc = document.getElementById('activeDatasetDesc');

    if (track === 'regression') {
        if (pill) pill.textContent = 'Selected: Track 1 · Bike Sharing Demand (Regression)';
        if (title) title.textContent = 'Bike Sharing Demand Forecasting';
        if (desc) desc.textContent = 'Configure temporal schedule and weather factors to evaluate predictions across all 10 trained regression algorithms.';
    } else if (track === 'classification') {
        if (pill) pill.textContent = 'Selected: Track 2 · Heart Disease Diagnosis (Classification)';
        if (title) title.textContent = 'Coronary Heart Disease Diagnostic Triage';
        if (desc) desc.textContent = 'Enter patient clinical, hemodynamic, and ECG indicators to evaluate diagnostic classifications across all 10 trained classification algorithms.';
    } else if (track === 'clustering') {
        if (pill) pill.textContent = 'Selected: Track 3 · Customer Segmentation (Clustering)';
        if (title) title.textContent = 'Customer Personality & Behavioral Segmentation';
        if (desc) desc.textContent = 'Input customer demographics, transaction channels, and spending to assign marketing personas across all 5 clustering configurations.';
    }

    // Render preset buttons
    renderPresetChips(track);

    // Update Chart Tab Labels for track
    updateChartTabLabels(track);
    appState.activeChartTab = 'multimetric';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Return from Workspace back to the 3-Dataset Selection screen
 */
function returnToDatasetSelection() {
    appState.currentDataset = null;
    appState.currentResults = null;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    document.getElementById('workspaceView').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('datasetSelectionView').style.display = 'block';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Render Quick Presets
 */
function renderPresetChips(track) {
    const container = document.getElementById('presetsList');
    container.innerHTML = '';

    const trackPresets = appState.presets[track] || {};
    const keys = Object.keys(trackPresets);

    if (keys.length === 0) {
        container.innerHTML = '<span class="field-hint">No presets available</span>';
        return;
    }

    keys.forEach(key => {
        const p = trackPresets[key];
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'preset-chip';
        chip.textContent = p.label;
        chip.title = p.desc || '';
        chip.onclick = () => applyPreset(track, key);
        container.appendChild(chip);
    });
}

/**
 * Apply Preset Values to Form
 */
function applyPreset(track, presetKey) {
    const preset = appState.presets[track]?.[presetKey];
    if (!preset || !preset.data) return;

    const data = preset.data;

    if (track === 'regression') {
        if ('season' in data) document.getElementById('reg_season').value = data.season;
        if ('yr' in data) document.getElementById('reg_yr').value = data.yr;
        if ('mnth' in data) document.getElementById('reg_mnth').value = data.mnth;
        if ('hr' in data) document.getElementById('reg_hr').value = data.hr;
        if ('workingday' in data) document.getElementById('reg_workingday').value = data.workingday;
        if ('weekday' in data) document.getElementById('reg_weekday').value = data.weekday;
        if ('holiday' in data) document.getElementById('reg_holiday').value = data.holiday;
        if ('weathersit' in data) document.getElementById('reg_weathersit').value = data.weathersit;
        if ('temp_c' in data) document.getElementById('reg_temp_c').value = data.temp_c;
        if ('atemp_c' in data) document.getElementById('reg_atemp_c').value = data.atemp_c;
        if ('hum' in data) document.getElementById('reg_hum').value = data.hum;
        if ('windspeed_kmh' in data) document.getElementById('reg_windspeed_kmh').value = data.windspeed_kmh;
    } else if (track === 'classification') {
        if ('age' in data) document.getElementById('clf_age').value = data.age;
        if ('sex' in data) document.getElementById('clf_sex').value = data.sex;
        if ('cp' in data) document.getElementById('clf_cp').value = data.cp;
        if ('trestbps' in data) document.getElementById('clf_trestbps').value = data.trestbps;
        if ('chol' in data) document.getElementById('clf_chol').value = data.chol;
        if ('fbs' in data) document.getElementById('clf_fbs').value = data.fbs;
        if ('restecg' in data) document.getElementById('clf_restecg').value = data.restecg;
        if ('thalch' in data) document.getElementById('clf_thalch').value = data.thalch;
        if ('exang' in data) document.getElementById('clf_exang').value = data.exang;
        if ('oldpeak' in data) document.getElementById('clf_oldpeak').value = data.oldpeak;
    } else if (track === 'clustering') {
        if ('Age' in data) document.getElementById('cl_age').value = data.Age;
        if ('Income' in data) document.getElementById('cl_income').value = data.Income;
        if ('Total_Spending' in data) document.getElementById('cl_spending').value = data.Total_Spending;
        if ('Children' in data) document.getElementById('cl_children').value = data.Children;
        if ('Recency' in data) document.getElementById('cl_recency').value = data.Recency;
        if ('Total_Purchases' in data) document.getElementById('cl_total_purchases').value = data.Total_Purchases;
        if ('NumStorePurchases' in data) document.getElementById('cl_store').value = data.NumStorePurchases;
        if ('NumWebPurchases' in data) document.getElementById('cl_web').value = data.NumWebPurchases;
        if ('NumCatalogPurchases' in data) document.getElementById('cl_catalog').value = data.NumCatalogPurchases;
        if ('NumWebVisitsMonth' in data) document.getElementById('cl_web_visits').value = data.NumWebVisitsMonth;
    }
}

/**
 * Handle Form Submission for any track
 */
async function handleFormSubmit(event, track) {
    event.preventDefault();

    let submitBtn;
    let url;
    let payload = {};

    if (track === 'regression') {
        submitBtn = document.getElementById('submitBtnRegression');
        url = '/api/predict/regression';
        payload = {
            season: parseInt(document.getElementById('reg_season').value),
            yr: parseInt(document.getElementById('reg_yr').value),
            mnth: parseInt(document.getElementById('reg_mnth').value),
            hr: parseInt(document.getElementById('reg_hr').value),
            workingday: parseInt(document.getElementById('reg_workingday').value),
            weekday: parseInt(document.getElementById('reg_weekday').value),
            holiday: parseInt(document.getElementById('reg_holiday').value),
            weathersit: parseInt(document.getElementById('reg_weathersit').value),
            temp_c: parseFloat(document.getElementById('reg_temp_c').value),
            atemp_c: parseFloat(document.getElementById('reg_atemp_c').value),
            hum: parseFloat(document.getElementById('reg_hum').value),
            windspeed_kmh: parseFloat(document.getElementById('reg_windspeed_kmh').value)
        };
    } else if (track === 'classification') {
        submitBtn = document.getElementById('submitBtnClassification');
        url = '/api/predict/classification';
        payload = {
            age: parseFloat(document.getElementById('clf_age').value),
            sex: document.getElementById('clf_sex').value,
            cp: document.getElementById('clf_cp').value,
            trestbps: parseFloat(document.getElementById('clf_trestbps').value),
            chol: parseFloat(document.getElementById('clf_chol').value),
            fbs: parseInt(document.getElementById('clf_fbs').value),
            restecg: document.getElementById('clf_restecg').value,
            thalch: parseFloat(document.getElementById('clf_thalch').value),
            exang: parseInt(document.getElementById('clf_exang').value),
            oldpeak: parseFloat(document.getElementById('clf_oldpeak').value)
        };
    } else if (track === 'clustering') {
        submitBtn = document.getElementById('submitBtnClustering');
        url = '/api/predict/clustering';
        payload = {
            Age: parseFloat(document.getElementById('cl_age').value),
            Income: parseFloat(document.getElementById('cl_income').value),
            Total_Spending: parseFloat(document.getElementById('cl_spending').value),
            Children: parseFloat(document.getElementById('cl_children').value),
            Recency: parseFloat(document.getElementById('cl_recency').value),
            Total_Purchases: parseFloat(document.getElementById('cl_total_purchases').value),
            NumStorePurchases: parseFloat(document.getElementById('cl_store').value),
            NumWebPurchases: parseFloat(document.getElementById('cl_web').value),
            NumCatalogPurchases: parseFloat(document.getElementById('cl_catalog').value),
            NumWebVisitsMonth: parseFloat(document.getElementById('cl_web_visits').value)
        };
    }

    const originalBtnText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Running All Algorithms...';

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!data.success) {
            alert('Error running models: ' + (data.error || 'Unknown error'));
            return;
        }

        appState.currentResults = data;
        displayResults(data, track);

    } catch (err) {
        console.error('Inference error:', err);
        alert('Failed to connect to the prediction server.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
}

/**
 * Update Chart Tab Labels based on selected track
 */
function updateChartTabLabels(track) {
    const tab1 = document.getElementById('tabChart1');
    const tab2 = document.getElementById('tabChart2');
    const tab3 = document.getElementById('tabChart3');
    if (!tab1 || !tab2 || !tab3) return;

    if (track === 'classification') {
        tab1.textContent = 'Multi-Metric Comparison (Cell 23)';
        tab2.textContent = 'Consolidated ROC Curves (Cell 24)';
        tab3.textContent = 'Live Patient Risk Prediction';
    } else if (track === 'regression') {
        tab1.textContent = 'R² Explained Variance (Cell 34)';
        tab2.textContent = 'Error Analysis (RMSE & MAE)';
        tab3.textContent = 'Live Demand Prediction';
    } else if (track === 'clustering') {
        tab1.textContent = 'Silhouette & Davies-Bouldin (Cell 19)';
        tab2.textContent = 'Calinski-Harabasz Variance Ratio';
        tab3.textContent = 'Live Cluster Membership Affinity';
    }
}

/**
 * Display outputs, champion banner, charts, and table
 */
function displayResults(data, track) {
    appState.currentResults = data;
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    const champion = data.champion;

    // 1. Champion Banner: Just algo name and the output it gave
    const modelNameEl = document.getElementById('championModelName');
    const answerValEl = document.getElementById('championAnswerValue');
    const answerUnitEl = document.getElementById('championAnswerUnit');
    const tierBadgeEl = document.getElementById('championTierBadge');
    const recEl = document.getElementById('championRecommendation');

    modelNameEl.textContent = champion.model_name;

    if (track === 'regression') {
        answerValEl.textContent = champion.predicted_value.toLocaleString();
        answerUnitEl.textContent = champion.unit;
        tierBadgeEl.textContent = champion.demand_tier;
        tierBadgeEl.style.backgroundColor = '#eff6ff';
        tierBadgeEl.style.color = '#1d4ed8';
        tierBadgeEl.style.borderColor = '#bfdbfe';
        recEl.textContent = `${champion.recommendation} (Forecasted for ${champion.hour_formatted} in ${champion.season_name} under ${champion.weather_desc} conditions).`;
    } else if (track === 'classification') {
        answerValEl.textContent = champion.disease_detected ? 'CAD Detected (Positive)' : 'No Significant CAD (Negative)';
        answerUnitEl.textContent = `${champion.probability_disease}% Probability`;
        tierBadgeEl.textContent = champion.risk_tier;
        
        if (champion.disease_detected) {
            tierBadgeEl.style.backgroundColor = '#fef2f2';
            tierBadgeEl.style.color = '#991b1b';
            tierBadgeEl.style.borderColor = '#fecaca';
        } else {
            tierBadgeEl.style.backgroundColor = '#f0fdf4';
            tierBadgeEl.style.color = '#166534';
            tierBadgeEl.style.borderColor = '#bbf7d0';
        }
        
        recEl.textContent = `${champion.clinical_advice} (Peak heart rate: ${champion.biomarker.actual_peak} bpm vs ${champion.biomarker.target_peak} bpm target).`;
    } else if (track === 'clustering') {
        answerValEl.textContent = champion.persona_name;
        answerUnitEl.textContent = champion.persona_badge;
        tierBadgeEl.textContent = `Segment ${champion.cluster_id} · Center Dist: ${champion.distance_to_center}`;
        tierBadgeEl.style.backgroundColor = '#fefce8';
        tierBadgeEl.style.color = '#854d0e';
        tierBadgeEl.style.borderColor = '#fef08a';
        recEl.textContent = `${champion.persona_desc} Recommended Strategy: ${champion.marketing_strategy}`;
    }

    // 2. Render Full Comparison Table directly reflecting notebook tables
    renderComparisonTable(data.all_algorithms, track);

    // 3. Update Chart Tabs and Render Chart
    updateChartTabLabels(track);
    renderChart(appState.activeChartTab || 'multimetric');

    // Smooth scroll down to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Render the full comparison table across all algorithms with complete notebook rubric metrics
 */
function renderComparisonTable(algorithms, track) {
    const thead = document.getElementById('algorithmsTableHead');
    const tbody = document.getElementById('algorithmsTableBody');
    const caption = document.getElementById('tableCaption');
    const title = document.getElementById('tableTitle');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (track === 'regression') {
        if (title) title.textContent = 'Consolidated 10-Algorithm Regression Comparison Table (Notebook Cell 34)';
        if (caption) caption.textContent = 'All 10 regression algorithms evaluated with live predictions alongside R², RMSE, and MAE benchmarks';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">Rank</th>
                    <th>Model Architecture</th>
                    <th>Predicted Demand (Live)</th>
                    <th>R² Score (↑)</th>
                    <th>RMSE (↓)</th>
                    <th>MAE (↓)</th>
                    <th>Evaluation Status</th>
                </tr>
            `;
        }
        algorithms.forEach(item => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${item.rank}</span></td>
                <td><strong>${item.model}</strong></td>
                <td><span class="table-num"><strong>${item.prediction_label}</strong></span></td>
                <td><span class="table-num font-mono"><strong>${item.r2 !== undefined ? item.r2.toFixed(4) : '-'}</strong></span></td>
                <td><span class="table-num font-mono">${item.rmse !== undefined ? item.rmse.toFixed(2) : '-'}</span></td>
                <td><span class="table-num font-mono">${item.mae !== undefined ? item.mae.toFixed(2) : '-'}</span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } else if (track === 'classification') {
        if (title) title.textContent = 'Consolidated 10-Algorithm Classification Comparison Table (Notebook Cell 22)';
        if (caption) caption.textContent = 'All 10 classification algorithms evaluated with live patient predictions alongside Accuracy, Precision, Recall, F1, and ROC-AUC';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">Rank</th>
                    <th>Model Architecture</th>
                    <th>Model Prediction (Live)</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 (weighted)</th>
                    <th>ROC-AUC</th>
                    <th>Evaluation Status</th>
                </tr>
            `;
        }
        algorithms.forEach(item => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            const isPos = item.prediction === 1 || (item.prediction_label && item.prediction_label.includes('Positive'));
            const callColor = isPos ? 'style="color:#b91c1c;font-weight:700;"' : 'style="color:#15803d;font-weight:700;"';
            const callLabel = isPos ? 'CAD Positive' : 'CAD Negative';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${item.rank}</span></td>
                <td><strong>${item.model}</strong></td>
                <td><span ${callColor}><strong>${item.probability_pct}%</strong> (${callLabel})</span></td>
                <td><span class="table-num font-mono">${item.accuracy !== undefined ? (item.accuracy * 100).toFixed(2) + '%' : '-'}</span></td>
                <td><span class="table-num font-mono">${item.precision !== undefined ? item.precision.toFixed(4) : '-'}</span></td>
                <td><span class="table-num font-mono">${item.recall !== undefined ? (item.recall * 100).toFixed(2) + '%' : '-'}</span></td>
                <td><span class="table-num font-mono"><strong>${item.f1 !== undefined ? item.f1.toFixed(4) : '-'}</strong></span></td>
                <td><span class="table-num font-mono">${item.roc_auc !== undefined ? item.roc_auc.toFixed(4) : '-'}</span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } else if (track === 'clustering') {
        if (title) title.textContent = 'Consolidated Clustering Evaluation Metrics Table (Notebook Cell 19)';
        if (caption) caption.textContent = 'All 5 clustering formulations evaluated with live persona assignment alongside Silhouette, Davies-Bouldin, and Calinski-Harabasz';
        if (thead) {
            thead.innerHTML = `
                <tr>
                    <th style="width: 50px;">#</th>
                    <th>Clustering Formulation</th>
                    <th>Clusters (k)</th>
                    <th>Assigned Persona (Live)</th>
                    <th>Silhouette Score (↑)</th>
                    <th>Davies-Bouldin (↓)</th>
                    <th>Calinski-Harabasz (↑)</th>
                    <th>Cluster Status</th>
                </tr>
            `;
        }
        algorithms.forEach((item, idx) => {
            const tr = document.createElement('tr');
            if (item.is_champion) tr.className = 'row-champion';

            tr.innerHTML = `
                <td><span class="rank-pill ${item.is_champion ? 'champion' : ''}">${idx + 1}</span></td>
                <td><strong>${item.name}</strong> <span class="text-muted" style="font-size:0.8rem;">(${item.linkage || 'k-means++'})</span></td>
                <td><span class="table-num font-mono">k = ${item.clusters}</span></td>
                <td><span class="table-num"><strong>${item.assignment}</strong></span></td>
                <td><span class="table-num font-mono"><strong>${item.silhouette !== undefined ? item.silhouette.toFixed(4) : '-'}</strong></span></td>
                <td><span class="table-num font-mono">${item.davies_bouldin !== undefined ? item.davies_bouldin.toFixed(4) : '-'}</span></td>
                <td><span class="table-num font-mono">${item.calinski_harabasz !== undefined ? item.calinski_harabasz.toFixed(2) : '-'}</span></td>
                <td><span class="status-tag ${item.is_champion ? 'champion' : ''}">${item.status}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }
}

/**
 * Switch Chart Tab
 */
function switchChartTab(tabKey) {
    appState.activeChartTab = tabKey;

    const tab1 = document.getElementById('tabChart1');
    const tab2 = document.getElementById('tabChart2');
    const tab3 = document.getElementById('tabChart3');

    [tab1, tab2, tab3].forEach(t => t && t.classList.remove('active'));

    if (tabKey === 'multimetric') {
        if (tab1) tab1.classList.add('active');
    } else if (tabKey === 'roc') {
        if (tab2) tab2.classList.add('active');
    } else {
        if (tab3) tab3.classList.add('active');
    }

    renderChart(tabKey);
}

/**
 * Render Chart using Chart.js directly matching Notebook Cells 23 & 24
 * Client-side Chart.js rendering engine — Hasini Battu (Person 3)
 * Responsible for: multi-model ROC curve rendering, appState chart lifecycle,
 * dataset switching, and dynamic legend/tooltip configuration.
 */
function renderChart(tabKey) {
    if (!appState.currentResults) return;

    const canvas = document.getElementById('comparisonChartCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const chartsData = appState.currentResults.charts;
    const track = appState.currentDataset;

    if (appState.chartInstance) {
        appState.chartInstance.destroy();
        appState.chartInstance = null;
    }

    const labels = chartsData.labels || [];

    // CLASSIFICATION CHARTS
    if (track === 'classification') {
        if (tabKey === 'multimetric') {
            // Notebook Cell 23: Multi-Metric Comparison Bar Chart (Accuracy, F1, ROC-AUC)
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Accuracy',
                            data: chartsData.accuracies,
                            backgroundColor: '#2563eb',
                            borderRadius: 4
                        },
                        {
                            label: 'F1-Score (weighted)',
                            data: chartsData.f1_scores,
                            backgroundColor: '#059669',
                            borderRadius: 4
                        },
                        {
                            label: 'ROC-AUC',
                            data: chartsData.roc_aucs,
                            backgroundColor: '#d97706',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Multi-Metric Comparison Bar Chart across All 10 Classification Algorithms (Notebook Cell 23)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' },
                            color: '#0f172a'
                        },
                        legend: { position: 'top' }
                    },
                    scales: {
                        y: {
                            min: 0.70,
                            max: 1.00,
                            title: { display: true, text: 'Metric Score (Held-out Test Split)' }
                        }
                    }
                }
            });
        } else if (tabKey === 'roc') {
            // Notebook Cell 24: Consolidated ROC Curves for All 10 Algorithms
            // ROC dataset builder & Chart.js line chart: Hasini Battu (Person 3)
            // Builds one dataset per model with AUC label; adds random-classifier diagonal
            const rocCurves = chartsData.roc_curves || [];
            const datasets = rocCurves.map(rc => ({
                label: `${rc.model} (AUC = ${rc.auc.toFixed(3)})`,
                data: rc.points,
                borderColor: rc.color,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3
            }));

            // Add Random Classifier diagonal baseline
            datasets.push({
                label: 'Random Classifier (AUC = 0.500)',
                data: [{x: 0, y: 0}, {x: 1, y: 1}],
                borderColor: '#94a3b8',
                borderDash: [5, 5],
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false
            });

            appState.chartInstance = new Chart(ctx, {
                type: 'line',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Consolidated Receiver Operating Characteristic (ROC) Curves — All 10 Algorithms (Notebook Cell 24)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' },
                            color: '#0f172a'
                        },
                        legend: { position: 'right', labels: { boxWidth: 12, font: { size: 10 } } }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            min: 0,
                            max: 1,
                            title: { display: true, text: 'False Positive Rate (1 - Specificity)' }
                        },
                        y: {
                            min: 0,
                            max: 1.02,
                            title: { display: true, text: 'True Positive Rate (Sensitivity / Recall)' }
                        }
                    }
                }
            });
        } else {
            // Tab 3: Live Patient Predictions across models
            const preds = chartsData.predictions || [];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Predicted CAD Probability (%) for Submitted Patient',
                        data: preds,
                        backgroundColor: preds.map((_, i) => i === 0 ? '#dc2626' : '#334155'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 0, max: 100, title: { display: true, text: 'Probability (%)' } }
                    }
                }
            });
        }
    } else if (track === 'regression') {
        if (tabKey === 'multimetric') {
            // R² Score Performance Comparison (Notebook Cell 34)
            const r2Data = chartsData.r2_scores || [];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'R² Score (Coefficient of Determination)',
                        data: r2Data,
                        backgroundColor: r2Data.map((_, i) => i === 0 ? '#d97706' : '#2563eb'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'R² Goodness-of-Fit Comparison across All 10 Regression Models (Notebook Cell 34)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: { min: 0.50, max: 1.00, title: { display: true, text: 'R² Score' } }
                    }
                }
            });
        } else if (tabKey === 'roc') {
            // Error Analysis: RMSE and MAE Comparison
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'RMSE (Root Mean Squared Error)',
                            data: chartsData.rmse_scores || [],
                            backgroundColor: '#dc2626',
                            borderRadius: 4
                        },
                        {
                            label: 'MAE (Mean Absolute Error)',
                            data: chartsData.mae_scores || [],
                            backgroundColor: '#059669',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Error Metric Comparison (RMSE vs MAE) across All 10 Models (Lower is Better)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Error in Bikes / Hour' } }
                    }
                }
            });
        } else {
            // Live Predicted Demand
            const preds = chartsData.predictions || [];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Predicted Demand (bikes/hr) for Submitted Inputs',
                        data: preds,
                        backgroundColor: preds.map((_, i) => i === 0 ? '#d97706' : '#334155'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { title: { display: true, text: 'Predicted Bikes / Hour' } }
                    }
                }
            });
        }
    } else if (track === 'clustering') {
        if (tabKey === 'multimetric') {
            // Silhouette Score & Davies-Bouldin Index (Notebook Cell 19)
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Silhouette Score (↑ Higher is Better)',
                            data: chartsData.silhouettes || [],
                            backgroundColor: '#059669',
                            borderRadius: 4
                        },
                        {
                            label: 'Davies-Bouldin Index (↓ Lower is Better)',
                            data: chartsData.davies_bouldins || [],
                            backgroundColor: '#2563eb',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Consolidated Clustering Evaluation: Silhouette Score & Davies-Bouldin Index (Notebook Cell 19)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Index Score' } }
                    }
                }
            });
        } else if (tabKey === 'roc') {
            // Calinski-Harabasz Variance Ratio
            const chData = chartsData.calinskis || [];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Calinski-Harabasz Variance Ratio Index (↑ Higher is Better)',
                        data: chData,
                        backgroundColor: chData.map((_, i) => i === 0 ? '#d97706' : '#475569'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Calinski-Harabasz Dispersion Criterion across Formulations (Notebook Cell 19)',
                            font: { family: "'Inter', sans-serif", size: 13, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Variance Ratio' } }
                    }
                }
            });
        } else {
            // Live Affinity
            const preds = chartsData.predictions || [];
            appState.chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Cluster Membership Affinity (%) for Submitted Profile',
                        data: preds,
                        backgroundColor: preds.map((_, i) => i === 0 ? '#d97706' : '#334155'),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 0, max: 100, title: { display: true, text: 'Affinity Score (%)' } }
                    }
                }
            });
        }
    }
}

/**
 * Scroll smoothly up to inputs
 */
function scrollToInputs() {
    const card = document.querySelector('.form-card');
    if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function openMetricsModal() {}
function closeMetricsModal() {}
function switchModalTab() {}

