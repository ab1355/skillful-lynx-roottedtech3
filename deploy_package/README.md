# AgentZero HR Analytics

This is a simple HR analytics application that provides performance predictions and team formation suggestions.

## Running the Application

To run the application, execute the following command:

```
./start.sh
```

The application will start and listen on the port specified by the PORT environment variable, or on port 8080 if not specified.

## API Endpoints

- GET /: Welcome message
- GET /dashboard: HR analytics dashboard data
- GET /health: Health check endpoint
- POST /predict_performance: Predict employee performance
- POST /form_team: Form an optimal team

For more details on how to use these endpoints, please refer to the API documentation.