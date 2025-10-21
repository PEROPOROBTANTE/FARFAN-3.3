let network;
let performanceChart;

async function loadDashboard() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        
        renderValidationGates(data.validation_results);
        renderContractCoverage(data.contract_coverage);
        renderCircuitBreakers(data.circuit_breaker_status);
        renderDependencyGraph(data.dependency_graph);
        renderCanaryGrid(data.canary_test_grid);
        renderPerformanceMetrics(data.performance_metrics);
        renderRemediationSuggestions(data.validation_results);
        
    } catch (error) {
        console.error('Dashboard load error:', error);
    }
}

function renderValidationGates(results) {
    const container = document.getElementById('validation-gates');
    const overallStatus = document.getElementById('overall-status');
    
    overallStatus.className = 'status-indicator ' + 
        (results.success ? 'status-pass' : 'status-fail');
    
    container.innerHTML = results.results.map(result => `
        <div class="metric">
            <span class="metric-label">
                <span class="status-indicator status-${result.passed ? 'pass' : 'fail'}"></span>
                ${result.gate_name}
            </span>
            <span class="metric-value">${result.status}</span>
        </div>
    `).join('');
}

function renderContractCoverage(coverage) {
    const container = document.getElementById('contract-coverage');
    container.innerHTML = `
        <div class="metric">
            <span class="metric-label">Total Methods</span>
            <span class="metric-value">${coverage.total_methods}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Valid Contracts</span>
            <span class="metric-value">${coverage.valid_contracts}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Missing Contracts</span>
            <span class="metric-value">${coverage.missing_contracts}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Coverage</span>
            <span class="metric-value">${coverage.coverage_percentage.toFixed(1)}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Fraction</span>
            <span class="metric-value">${coverage.coverage_fraction}</span>
        </div>
    `;
}

function renderCircuitBreakers(status) {
    const container = document.getElementById('circuit-breaker-status');
    container.innerHTML = Object.entries(status).map(([adapter, info]) => `
        <div class="circuit-item circuit-${info.state.toLowerCase()}">
            <h3>${adapter.replace('_', ' ')}</h3>
            <div class="state">${info.state}</div>
            <div style="font-size: 11px; margin-top: 8px;">
                Success: ${(info.success_rate * 100).toFixed(1)}%<br>
                Failures: ${info.recent_failures}
            </div>
        </div>
    `).join('');
}

function renderDependencyGraph(graphData) {
    const container = document.getElementById('dependency-graph');
    
    const nodes = new vis.DataSet(graphData.nodes.map(node => ({
        id: node.id,
        label: node.label + '\n(' + node.method_count + ' methods)',
        color: {
            background: node.color,
            border: node.color,
            highlight: { background: node.color, border: '#ffffff' }
        },
        font: { color: '#ffffff', size: 14 },
        shape: 'box',
        title: `${node.label}\nStatus: ${node.status}\nCircuit: ${node.circuit_state}`
    })));
    
    const edges = new vis.DataSet(graphData.edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        arrows: 'to',
        color: { color: edge.color, highlight: edge.color },
        width: edge.impact === 'propagating' ? 3 : 1,
        title: `Binding: ${edge.from} → ${edge.to}\nImpact: ${edge.impact}`
    })));
    
    const data = { nodes, edges };
    const options = {
        physics: {
            enabled: true,
            barnesHut: { gravitationalConstant: -8000, springLength: 200 }
        },
        layout: { improvedLayout: true },
        interaction: { hover: true }
    };
    
    network = new vis.Network(container, data, options);
}

function renderCanaryGrid(grid) {
    const container = document.getElementById('canary-grid');
    const cells = [];
    
    for (const [adapter, methods] of Object.entries(grid)) {
        for (const [method, status] of Object.entries(methods)) {
            cells.push(`
                <div class="canary-cell canary-${status}" 
                     onclick="handleCanaryClick('${adapter}', '${method}', '${status}')"
                     title="${adapter}.${method}">
                    ${method.substring(0, 10)}
                </div>
            `);
        }
    }
    
    container.innerHTML = cells.join('') || '<p>No canary tests available</p>';
}

function renderPerformanceMetrics(metrics) {
    const ctx = document.getElementById('performance-chart');
    
    const adapters = Object.keys(metrics);
    const p50Data = adapters.map(a => metrics[a].p50_latency);
    const p95Data = adapters.map(a => metrics[a].p95_latency);
    const p99Data = adapters.map(a => metrics[a].p99_latency);
    
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: adapters.map(a => a.replace('_', ' ')),
            datasets: [
                {
                    label: 'P50 Latency (ms)',
                    data: p50Data,
                    backgroundColor: 'rgba(88, 166, 255, 0.5)',
                    borderColor: 'rgba(88, 166, 255, 1)',
                    borderWidth: 1
                },
                {
                    label: 'P95 Latency (ms)',
                    data: p95Data,
                    backgroundColor: 'rgba(255, 193, 7, 0.5)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 1
                },
                {
                    label: 'P99 Latency (ms)',
                    data: p99Data,
                    backgroundColor: 'rgba(220, 53, 69, 0.5)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    beginAtZero: true,
                    ticks: { color: '#8b949e' },
                    grid: { color: '#30363d' }
                },
                x: { 
                    ticks: { color: '#8b949e' },
                    grid: { color: '#30363d' }
                }
            },
            plugins: {
                legend: { labels: { color: '#c9d1d9' } }
            }
        }
    });
}

function renderRemediationSuggestions(results) {
    const container = document.getElementById('remediation-suggestions');
    const suggestions = [];
    
    for (const result of results.results) {
        if (result.metrics && result.metrics.remediation_suggestions) {
            suggestions.push(...result.metrics.remediation_suggestions);
        }
    }
    
    if (suggestions.length === 0) {
        container.innerHTML = '<p style="color: #28a745;">✓ All checks passed. No remediation needed.</p>';
        return;
    }
    
    container.innerHTML = suggestions.map(s => `
        <div class="remediation-box">
            <h4>${s.error_code}</h4>
            <p>${s.suggested_fix}</p>
            ${s.command ? `<p style="margin-top: 8px;"><code>${s.command}</code></p>` : ''}
        </div>
    `).join('');
}

async function handleCanaryClick(adapter, method, status) {
    if (status === 'fail') {
        if (confirm(`Rebaseline ${adapter}.${method}?`)) {
            try {
                const response = await fetch(`/api/canary_rebaseline/${adapter}/${method}`, {
                    method: 'POST'
                });
                const result = await response.json();
                alert(result.message || result.error);
                refreshDashboard();
            } catch (error) {
                alert('Rebaseline failed: ' + error);
            }
        }
    }
}

async function refreshDashboard() {
    await loadDashboard();
}

loadDashboard();
setInterval(refreshDashboard, 30000);
