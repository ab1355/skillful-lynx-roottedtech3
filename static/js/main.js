let performanceChart, teamChart;

document.getElementById('performance-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = {
        age: parseInt(document.getElementById('age').value),
        years_at_company: parseInt(document.getElementById('years_at_company').value),
        years_in_current_role: parseInt(document.getElementById('years_in_current_role').value),
        job_satisfaction: parseInt(document.getElementById('job_satisfaction').value),
        job_involvement: parseInt(document.getElementById('job_involvement').value),
        relationship_satisfaction: parseInt(document.getElementById('relationship_satisfaction').value),
        work_life_balance: parseInt(document.getElementById('work_life_balance').value)
    };
    
    fetch('/predict_performance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
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
        document.getElementById('performance-error').textContent = 'An error occurred. Please try again.';
    });
});

document.getElementById('team-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = {
        team_size: parseInt(document.getElementById('team_size').value),
        required_skills: document.getElementById('required_skills').value.split(',').map(skill => skill.trim())
    };
    
    fetch('/form_team', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
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
        document.getElementById('team-error').textContent = 'An error occurred. Please try again.';
    });
});

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
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5
                }
            }
        }
    });
}