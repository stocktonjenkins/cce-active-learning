from probcal.lib.observer import IObserver, ISubject
from probcal.active_learning.active_learning_types import ActiveLearningEvaluationResults, ModelAccuracyResults


class ActiveLearningModelAccuracyLogger(IObserver[ActiveLearningEvaluationResults]):
    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        # Get the current state from the subject
        state = subject.get_state()
        model_accuracy_results = state.model_accuracy_results

        # Calculate model accuracy results

        # Print the results
        print(f"Accuracy: {model_accuracy_results.accuracy}")
        print(f"Precision: {model_accuracy_results.precision}")
        print(f"Recall: {model_accuracy_results.recall}")
        print(f"F1 Score: {model_accuracy_results.f1_score}")