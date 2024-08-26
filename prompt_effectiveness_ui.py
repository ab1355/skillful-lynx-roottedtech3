import pandas as pd
from risk_prediction_refinement import AdvancedRiskPredictor

def get_predefined_input():
    return pd.DataFrame({
        'prompt_length': [250],
        'prompt_complexity': [7],
        'target_task_difficulty': [8],
        'model_size': ['large'],
        'context_relevance': [0.9],
        'instruction_clarity': [9],
        'expected_output_length': [500],
        'domain_specificity': [6],
        'creativity_required': [8],
        'time_constraint': [30]
    })

def provide_suggestions(predictions, user_input):
    suggestions = []

    if predictions['accuracy_score'] < 70:
        if user_input['instruction_clarity'].values[0] < 8:
            suggestions.append("Consider improving the clarity of instructions to increase accuracy.")
        if user_input['context_relevance'].values[0] < 0.8:
            suggestions.append("Try to provide more relevant context to improve accuracy.")

    if predictions['creativity_score'] < 70:
        if user_input['creativity_required'].values[0] > 7 and user_input['time_constraint'].values[0] < 20:
            suggestions.append("Allow more time for tasks requiring high creativity.")
        if user_input['domain_specificity'].values[0] > 8:
            suggestions.append("Consider broadening the domain to encourage more creative responses.")

    if predictions['relevance_score'] < 70:
        if user_input['prompt_complexity'].values[0] > 8:
            suggestions.append("Try simplifying the prompt to improve relevance.")
        if user_input['target_task_difficulty'].values[0] > 8 and user_input['instruction_clarity'].values[0] < 9:
            suggestions.append("For difficult tasks, ensure instructions are very clear to maintain relevance.")

    return suggestions

def main():
    print("Welcome to the AgentZero Prompt Effectiveness Predictor!")
    print("Loading the model...")
    risk_predictor = AdvancedRiskPredictor()
    data = risk_predictor.load_llm_prompting_data()
    risk_predictor.train_and_evaluate(data)
    print("Model loaded and trained successfully.")

    user_input = get_predefined_input()
    predictions = risk_predictor.predict_prompt_effectiveness(user_input)

    print("\nPredicted prompt effectiveness scores:")
    for key, value in predictions.items():
        print(f"{key}: {value:.2f}")

    suggestions = provide_suggestions(predictions, user_input)
    if suggestions:
        print("\nSuggestions for improvement:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    else:
        print("\nYour prompt parameters look good! No specific suggestions for improvement.")

    print("Thank you for using the AgentZero Prompt Effectiveness Predictor!")

if __name__ == "__main__":
    main()