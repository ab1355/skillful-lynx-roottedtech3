from flask import Flask, render_template, request, jsonify
import asyncio
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import json
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

config = Config()
agents = [Agent(f"Agent_{i}", "classification", "classification") for i in range(config.num_initial_agents)]
mas = MultiAgentSystem(agents, config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    num_steps = int(request.form.get('num_steps', 100))
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    final_performance = loop.run_until_complete(mas.run_simulation(num_steps))
    
    performance_history = mas.performance_history
    workload_history = mas.workload_history
    
    plt.figure(figsize=(10, 5))
    plt.plot(performance_history, label='Performance')
    plt.plot(workload_history, label='Workload')
    plt.legend()
    plt.title('System Performance and Workload')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({
        'final_performance': final_performance,
        'plot': plot_url,
        'num_agents': len(mas.agents),
        'specialization_changes': len(mas.specialization_changes)
    })

if __name__ == '__main__':
    app.run(debug=True)