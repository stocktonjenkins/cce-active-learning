from probcal.active_learning.types import ActiveLearningEvaluationResults
from probcal.lib.observer import IObserver, ISubject


class ActiveLearningAverageCCE(IObserver):
    def update(self, subject: ISubject[ActiveLearningEvaluationResults]) -> None:
        # do something with subject.get_state()
        print(subject.get_state().calibration_results)
