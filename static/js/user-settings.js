class UserSettings {
    constructor() {
        this.settings = JSON.parse(localStorage.getItem('userSettings')) || {
            preferredChartType: 'bar',
            defaultView: 'performance',
            notificationsEnabled: true
        };
    }

    saveSettings() {
        localStorage.setItem('userSettings', JSON.stringify(this.settings));
    }

    getSettings() {
        return this.settings;
    }

    updateSetting(key, value) {
        this.settings[key] = value;
        this.saveSettings();
    }
}

const userSettings = new UserSettings();

function updateChartType(chartType) {
    userSettings.updateSetting('preferredChartType', chartType);
    updateCharts();
}

function updateDefaultView(view) {
    userSettings.updateSetting('defaultView', view);
    showDefaultView();
}

function toggleNotifications(enabled) {
    userSettings.updateSetting('notificationsEnabled', enabled);
}

function showDefaultView() {
    const defaultView = userSettings.getSettings().defaultView;
    // Hide all views
    document.querySelectorAll('.dashboard-card').forEach(card => card.style.display = 'none');
    // Show the default view
    document.getElementById(`${defaultView}-card`).style.display = 'block';
}

function updateCharts() {
    const chartType = userSettings.getSettings().preferredChartType;
    if (performanceChart) {
        performanceChart.config.type = chartType;
        performanceChart.update();
    }
    if (teamChart) {
        teamChart.config.type = chartType;
        teamChart.update();
    }
}

// Initialize user settings when the page loads
document.addEventListener('DOMContentLoaded', function() {
    showDefaultView();
    updateCharts();
});