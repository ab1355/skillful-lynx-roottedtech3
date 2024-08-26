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
});

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

function validateInput(event) {
    const input = event.target;
    const errorElement = input.nextElementSibling;
    
    if (input.validity.valid) {
        if (errorElement && errorElement.classList.contains('error-message')) {
            errorElement.remove();
        }
    } else {
        if (!errorElement || !errorElement.classList.contains('error-message')) {
            const errorMessage = document.createElement('div');
            errorMessage.className = 'error-message';
            errorMessage.textContent = input.validationMessage;
            input.parentNode.insertBefore(errorMessage, input.nextSibling);
        }
    }
}

function handlePerformanceFormSubmit(e) {
    e.preventDefault();
    const form = e.target;
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const formData = {
        age: parseInt(document.getElementById('age').value),
        years_at_company: parseInt(document.getElementById('years_at_company').value),
        years_in_current_role: parseInt(document.getElementById('years_in_current_role').value),
        job_satisfaction: parseInt(document.getElementById('job_satisfaction').value),
        job_involvement: parseInt(document.getElementById('job_involvement').value),
        relationship_satisfaction: parseInt(document.getElementById('relationship_satisfaction').value),
        work_life_balance: parseInt(document.getElementById('work_life_balance').value)
    };
    
    const spinner = document.getElementById('performance-spinner');
    spinner.style.display = 'block';

    fetch('/predict_performance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        spinner.style.display = 'none';
        if (data.error) {
            document.getElementById('performance-error').textContent = data.error;
            document.getElementById('performance-result').innerHTML = '';
        } else {
            document.getElementById('performance-error').textContent = '';
            document.getElementById('performance-result').innerHTML = `
                <h3>Predicted Performance: ${data.predicted_performance.toFixed(2)}</h3>
            `;
            updatePerformanceChart(data.predicted_performance);
        }
    })
    .catch(error => {
        spinner.style.display = 'none';
        document.getElementById('performance-error').textContent = 'An error occurred. Please try again.';
    });
}

function handleTeamFormSubmit(e) {
    e.preventDefault();
    const form = e.target;
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const formData = {
        team_size: parseInt(document.getElementById('team_size').value),
        required_skills: document.getElementById('required_skills').value.split(',').map(skill => skill.trim())
    };
    
    const spinner = document.getElementById('team-spinner');
    spinner.style.display = 'block';

    fetch('/form_team', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        spinner.style.display = 'none';
        if (data.error) {
            document.getElementById('team-error').textContent = data.error;
            document.getElementById('team-result').innerHTML = '';
        } else {
            document.getElementById('team-error').textContent = '';
            let teamHtml = '<h3>Optimal Team:</h3>';
            data.optimal_team.forEach((member, index) => {
                teamHtml += `
                    <h4>Member ${index + 1}</h4>
                    <p>Age: ${member.age}</p>
                    <p>Department: ${member.department}</p>
                    <p>Job Role: ${member.job_role}</p>
                    <p>Performance Score: ${member.performance_score.toFixed(2)}</p>
                    <p>Job Satisfaction: ${member.job_satisfaction}</p>
                    <p>Years at Company: ${member.years_at_company}</p>
                    <hr>
                `;
            });
            document.getElementById('team-result').innerHTML = teamHtml;
            updateTeamChart(data.optimal_team);
        }
    })
    .catch(error => {
        spinner.style.display = 'none';
        document.getElementById('team-error').textContent = 'An error occurred. Please try again.';
    });
}

function updatePerformanceChart(performance) {
    const ctx = document.getElementById('performance-chart').getContext('2d');
    if (performanceChart) {
        performanceChart.destroy();
    }
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Performance'],
            datasets: [{
                label: 'Performance Score',
                data: [performance],
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
            }
        }
    });
}

function updateTeamChart(team) {
    const ctx = document.getElementById('team-chart').getContext('2d');
    if (teamChart) {
        teamChart.destroy();
    }
    teamChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: team.map((_, index) => `Member ${index + 1}`),
            datasets: [{
                label: 'Performance Score',
                data: team.map(member => member.performance_score),
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
            }
        }
    });
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}