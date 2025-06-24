from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature names based on your data
features_list = [
    ('speed', 'Speed (km/h)', 0, 200),
    ('brake_pressure', 'Brake Pressure (psi)', 0, 1000),
    ('brake_temp', 'Brake Temperature (°C)', 0, 500),
    ('brake_fluid_level', 'Brake Fluid Level (0-1)', 0, 1),
    ('engine_oil_temp', 'Engine Oil Temp (°C)', 0, 150),
    ('engine_oil_pressure', 'Engine Oil Pressure (bar)', 0, 10),
    ('abs_status', 'ABS Status (0=off, 1=on)', 0, 1),
    ('tire_pressure', 'Tire Pressure (psi)', 0, 50),
    ('vehicle_weight', 'Vehicle Weight (kg)', 0, 5000),
    ('time_since_last_maintenance', 'Time Since Last Maintenance (days)', 0, 365)
]

# HTML + CSS template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Brake Failure Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --danger: #ef233c;
            --success: #2ecc71;
            --dark: #2b2d42;
            --light: #f8f9fa;
            --gray: #adb5bd;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f2f5;
            color: var(--dark);
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h2 {
            color: var(--primary);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: var(--gray);
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--dark);
        }
        
        input[type=number] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
            transition: border 0.3s ease;
        }
        
        input[type=number]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
        }
        
        .btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            margin-top: 1rem;
            width: 100%;
        }
        
        .btn:hover {
            background-color: #3a56d4;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
        }
        
        .result-container {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            animation: fadeIn 0.5s ease;
        }
        
        .failure {
            background-color: rgba(239, 35, 60, 0.1);
            border-left: 4px solid var(--danger);
        }
        
        .safe {
            background-color: rgba(46, 204, 113, 0.1);
            border-left: 4px solid var(--success);
        }
        
        .result-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .result-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .result-desc {
            color: var(--gray);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 1rem;
                padding: 1.5rem;
            }
            
            .header h2 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🚗 Brake Failure Prediction System</h2>
            <p>Enter vehicle parameters to assess brake failure risk</p>
        </div>
        
        <form action="/predict" method="post">
            <div class="form-grid">
                {% for name, label, min_val, max_val in features %}
                <div class="form-group">
                    <label for="{{ name }}">{{ label }}</label>
                    <input type="number" 
                           id="{{ name }}" 
                           name="{{ name }}" 
                           step="any" 
                           min="{{ min_val }}" 
                           max="{{ max_val }}"
                           placeholder="Enter {{ label.split('(')[0].strip() }}"
                           required>
                </div>
                {% endfor %}
            </div>
            
            <button type="submit" class="btn">Analyze Brake System</button>
        </form>
        
        {% if result %}
        <div class="result-container {{ 'failure' if 'Likely' in result else 'safe' }}">
            <div class="result-icon">
                {% if 'Likely' in result %}⚠️{% else %}✅{% endif %}
            </div>
            <div class="result-title">{{ result.split('!')[0] if '!' in result else result }}</div>
            <div class="result-desc">
                {% if 'Likely' in result %}
                Immediate inspection recommended
                {% else %}
                System appears to be functioning normally
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template, features=features_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[name]) for name, _, _, _ in features_list]
        prediction = model.predict([input_data])[0]
        result = "🚨 Brake Failure Likely!" if prediction == 1 else "✅ No Brake Failure Detected"
        return render_template_string(html_template, result=result, features=features_list)
    except Exception as e:
        return render_template_string(html_template, result=f"Error: {str(e)}", features=features_list)

if __name__ == '__main__':
    app.run(debug=True, port=5001)