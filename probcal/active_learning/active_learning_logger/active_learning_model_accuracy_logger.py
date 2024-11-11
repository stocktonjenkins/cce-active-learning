from probcal.lib.observer import IObserver, ISubject
from probcal.active_learning.active_learning_types import ActiveLearningEvaluationResults, ModelAccuracyResults

class ActiveLearningModelAccuracyLogger(IObserver[ActiveLearningEvaluationResults]):
    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        # Get the current state from the subject
        state = subject.get_state()
        
        # Assuming state contains y_true and y_pred
        y_true = state.y_true
        y_pred = state.y_pred
        
        # Calculate model accuracy results
        model_accuracy_results = ModelAccuracyResults.from_predictions(y_true, y_pred)
        
        # Print the results
        print(f"Accuracy: {model_accuracy_results.accuracy}")
        print(f"Precision: {model_accuracy_results.precision}")
        print(f"Recall: {model_accuracy_results.recall}")
        print(f"F1 Score: {model_accuracy_results.f1_score}")