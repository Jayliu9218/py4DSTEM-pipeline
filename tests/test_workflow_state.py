import unittest

from app.services.workflow_state import WorkflowState, WorkflowStep


class WorkflowStateTests(unittest.TestCase):
    def test_bragg_parameter_change_marks_completed_downstream_results_stale(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.BRAGG_FULL)
        state.mark_completed(WorkflowStep.CALIBRATION_APPLY)
        state.mark_completed(WorkflowStep.ORIENTATION_MATCH)
        state.mark_completed(WorkflowStep.STRAIN_MAP)

        state.parameters_updated(WorkflowStep.BRAGG_FULL)

        self.assertTrue(state.is_stale(WorkflowStep.BRAGG_FULL))
        self.assertTrue(state.is_stale(WorkflowStep.CALIBRATION_APPLY))
        self.assertTrue(state.is_stale(WorkflowStep.ORIENTATION_MATCH))
        self.assertTrue(state.is_stale(WorkflowStep.STRAIN_MAP))

    def test_recomputing_a_step_clears_only_that_step(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.BRAGG_FULL)
        state.mark_completed(WorkflowStep.CALIBRATION_APPLY)
        state.parameters_updated(WorkflowStep.BRAGG_FULL)

        state.mark_completed(WorkflowStep.BRAGG_FULL)

        self.assertFalse(state.is_stale(WorkflowStep.BRAGG_FULL))
        self.assertTrue(state.is_stale(WorkflowStep.CALIBRATION_APPLY))

    def test_upstream_execution_invalidates_existing_downstream_result(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.ORIENTATION_PLAN)
        state.mark_completed(WorkflowStep.ORIENTATION_MATCH)

        state.mark_completed(WorkflowStep.ORIENTATION_PLAN)

        self.assertTrue(state.is_stale(WorkflowStep.ORIENTATION_MATCH))


if __name__ == "__main__":
    unittest.main()
