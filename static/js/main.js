let performanceChart, teamChart;

document.addEventListener('DOMContentLoaded', function() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    darkModeToggle.addEventListener('click', toggleDarkMode);

    const performanceForm = document.getElementById('performance-form');
    const teamForm = document.getElementById('team-form');

    performanceForm.addEventListener('submit', handlePerformanceFormSubmit);
    teamForm.addEventListener('submit', handleTeamFormSubmit);

    // Client-side form validation
    performanceForm.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', validateInput);
    });

    teamForm.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', validateInput);
    });

    // Initialize charts
    initializeCharts();

    // Add event listeners for chart type selection
    document.getElementById('chartTypeSelector').addEventListener('change', function(e) {
        updateChartType(e.target.value);
    });

    // Add event listeners for data filtering
    document.getElementById('dataFilter').addEventListener('change', function(e) {
        filterChartData(e.target.value);
    });
});

function initializeCharts() {
    const performanceCtx = document.getElementById('performance-chart').getContext('2d');
    const teamCtx = document.getElementById('team-chart').getContext('2d');

    performanceChart = new Chart(performanceCtx, {
        type: userSettings.getSettings().preferredChartType,
        data: {
            labels: ['Predicted Performance'],
            datasets: [{
                label: 'Performance Score',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Score: ${context.raw.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });

    teamChart = new Chart(teamCtx, {
        type: userSettings.getSettings().preferredChartType,
        data: {
            labels: [],
            datasets: [{
                label: 'Performance Score',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const member = context.chart.data.datasets[0].data[context.dataIndex];
                            return [
                                `Score: ${member.performance_score.toFixed(2)}`,
                                `Age: ${member.age}`,
                                `Department: ${member.department}`,
                                `Job Role: ${member.job_role}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

function updateChartType(chartType) {
    userSettings.updateSetting('preferredChartType', chartType);
    performanceChart.config.type = chartType;
    teamChart.config.type = chartType;
    performanceChart.update();
    teamChart.update();
}

function filterChartData(filter) {
    // Implement data filtering logic here
    // For example, you could filter team members by department
    const filteredTeam = teamData.filter(member => member.department === filter);
    updateTeamChart(filteredTeam);
}

// ... (rest of the existing code)

function updatePerformanceChart(performance) {
    performanceChart.data.datasets[0].data = [performance];
    performanceChart.update();
}

function updateTeamChart(team) {
    teamChart.data.labels = team.map((_, index) => `Member ${index + 1}`);
    teamChart.data.datasets[0].data = team;
    teamChart.update();
}

// ... (rest of the existing code)// Test comment
