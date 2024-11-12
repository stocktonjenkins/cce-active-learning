from probcal.active_learning.active_learning_types import ActiveLearningEvaluationResults
from probcal.lib.observer import IObserver, ISubject
import csv


class ActiveLearningModelAccuracyLogger(IObserver[ActiveLearningEvaluationResults]):
    def __init__(self, log_file: str):
        self.log_file = log_file
        # Initialize the CSV file with headers
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Accuracy", "Precision", "Recall", "F1 Score"])

    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        # Get the current state from the subject
        state = subject.get_state()
        
        # Log the results to the CSV file
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                state.model_accuracy_results.accuracy,
                state.model_accuracy_results.precision,
                state.model_accuracy_results.recall,
                state.model_accuracy_results.f1_score
            ])

        # Print the results to the terminal
        print(f"Accuracy: {state.model_accuracy_results.accuracy}")
        print(f"Precision: {state.model_accuracy_results.precision}")
        print(f"Recall: {state.model_accuracy_results.recall}")
        print(f"F1 Score: {state.model_accuracy_results.f1_score}")
