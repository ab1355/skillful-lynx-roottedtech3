import http.server
import socketserver
import json
import logging
from urllib.parse import urlparse
from hr_data import load_hr_data, calculate_avg_performance, get_dept_distribution, get_age_distribution, get_tenure_distribution
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')

# Load HR data
hr_data = load_hr_data(config.DATA_FILE)

def predict_performance(employee_data):
    if 'performance_score' not in employee_data or 'years_at_company' not in employee_data:
        raise ValueError("Missing required fields: performance_score and years_at_company")
    
    base_score = float(employee_data["performance_score"])
    years = int(employee_data["years_at_company"])
    
    if not (config.MIN_PERFORMANCE_SCORE <= base_score <= config.MAX_PERFORMANCE_SCORE):
        raise ValueError(f"Performance score must be between {config.MIN_PERFORMANCE_SCORE} and {config.MAX_PERFORMANCE_SCORE}")
    if years < 0:
        raise ValueError("Years at company cannot be negative")
    
    years_factor = min(years * 0.01, 0.2)
    return round(min(base_score + years_factor, config.MAX_PERFORMANCE_SCORE), 2)

def form_optimal_team(team_size, required_skills):
    if not isinstance(team_size, int) or team_size <= 0 or team_size > config.MAX_TEAM_SIZE:
        raise ValueError(f"Team size must be a positive integer not exceeding {config.MAX_TEAM_SIZE}")
    if not isinstance(required_skills, list) or not required_skills:
        raise ValueError("Required skills must be a non-empty list")
    
    suitable_employees = [emp for emp in hr_data if any(skill in emp["skills"] for skill in required_skills)]
    suitable_employees.sort(key=lambda x: x["performance_score"], reverse=True)
    return [{"id": emp["id"], "name": emp["name"], "skills": emp["skills"]} for emp in suitable_employees[:team_size]]

class AgentZeroHandler(http.server.SimpleHTTPRequestHandler):
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        try:
            parsed_path = urlparse(self.path)
            if parsed_path.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Welcome to AgentZero HR Analytics")
            elif parsed_path.path == '/dashboard':
                dashboard_data = {
                    "total_employees": len(hr_data),
                    "avg_performance": calculate_avg_performance(hr_data),
                    "dept_distribution": get_dept_distribution(hr_data),
                    "age_distribution": get_age_distribution(hr_data),
                    "tenure_distribution": get_tenure_distribution(hr_data)
                }
                self.send_json_response(dashboard_data)
            elif parsed_path.path == '/health':
                self.send_json_response({"status": "healthy"})
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            logging.error(f"Error in GET request: {str(e)}")
            self.send_error(500, "Internal Server Error")

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            if self.path == '/predict_performance':
                predicted_performance = predict_performance(data)
                self.send_json_response({'predicted_performance': predicted_performance})
            elif self.path == '/form_team':
                if 'team_size' not in data or 'required_skills' not in data:
                    raise ValueError('Missing required fields: team_size and required_skills')
                optimal_team = form_optimal_team(data["team_size"], data["required_skills"])
                self.send_json_response({'optimal_team': optimal_team})
            else:
                self.send_error(404, "File not found")
        except ValueError as ve:
            logging.warning(f"Invalid input: {str(ve)}")
            self.send_json_response({"error": str(ve)}, 400)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON in request body")
            self.send_json_response({"error": "Invalid JSON in request body"}, 400)
        except Exception as e:
            logging.error(f"Error in POST request: {str(e)}")
            self.send_error(500, "Internal Server Error")

if __name__ == '__main__':
    Handler = AgentZeroHandler

    with socketserver.TCPServer((config.HOST, config.PORT), Handler) as httpd:
        logging.info(f"Serving at {config.HOST}:{config.PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logging.info("Server stopped by user")
        finally:
            httpd.server_close()
            logging.info("Server closed")