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

    def test_strain_requires_braggvectors_not_calibration_completion(self) -> None:
        state = WorkflowState()
        self.assertIn("bragg full", state.prerequisite_message([WorkflowStep.BRAGG_FULL]))

        state.mark_completed(WorkflowStep.BRAGG_FULL)

        self.assertEqual(state.prerequisite_message([WorkflowStep.BRAGG_FULL]), "")
        self.assertFalse(state.is_completed(WorkflowStep.CALIBRATION_APPLY))

    def test_dataset_role_change_marks_completed_workflow_stale(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.BRAGG_FULL)
        state.mark_completed(WorkflowStep.CALIBRATION_APPLY)
        state.mark_completed(WorkflowStep.ORIENTATION_MATCH)
        state.mark_completed(WorkflowStep.STRAIN_MAP)

        state.set_dataset_role("target_datacube", "/data")

        self.assertEqual(state.dataset_roles.target_datacube, "/data")
        self.assertTrue(state.is_stale(WorkflowStep.BRAGG_FULL))
        self.assertTrue(state.is_stale(WorkflowStep.CALIBRATION_APPLY))
        self.assertTrue(state.is_stale(WorkflowStep.ORIENTATION_MATCH))
        self.assertTrue(state.is_stale(WorkflowStep.STRAIN_MAP))

    def test_dataset_role_can_be_overwritten_and_cleared(self) -> None:
        state = WorkflowState()

        state.set_dataset_role("vacuum_probe", "/probe_a")
        state.set_dataset_role("vacuum_probe", "/probe_b")
        state.set_dataset_role("vacuum_probe", None)

        self.assertIsNone(state.dataset_roles.vacuum_probe)

    def test_bf_df_is_recommended_but_does_not_invalidate_dpc(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.DPC)

        state.parameters_updated(WorkflowStep.BF_DF_PREVIEW)

        self.assertFalse(state.is_stale(WorkflowStep.DPC))

    def test_target_data_change_invalidates_dpc_chain(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.DPC)
        state.mark_completed(WorkflowStep.PARALLAX)

        state.set_dataset_role("target_datacube", "/new-datacube")

        self.assertTrue(state.is_stale(WorkflowStep.DPC))
        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX))

    def test_dpc_stage_dependencies_follow_scientific_gates(self) -> None:
        state = WorkflowState()
        for step in (
            WorkflowStep.DPC_SEGMENTED,
            WorkflowStep.DPC_PREPROCESS,
            WorkflowStep.DPC_REVIEW,
            WorkflowStep.DPC,
        ):
            state.mark_completed(step)

        state.parameters_updated(WorkflowStep.DPC_SEGMENTED)
        self.assertTrue(state.is_stale(WorkflowStep.DPC_SEGMENTED))
        self.assertFalse(state.is_stale(WorkflowStep.DPC_PREPROCESS))
        self.assertFalse(state.is_stale(WorkflowStep.DPC_REVIEW))
        self.assertFalse(state.is_stale(WorkflowStep.DPC))

        state.parameters_updated(WorkflowStep.DPC_PREPROCESS)
        self.assertTrue(state.is_stale(WorkflowStep.DPC_PREPROCESS))
        self.assertTrue(state.is_stale(WorkflowStep.DPC_REVIEW))
        self.assertTrue(state.is_stale(WorkflowStep.DPC))

    def test_parallax_stage_dependencies_follow_acceptance_gates(self) -> None:
        state = WorkflowState()
        for step in (
            WorkflowStep.PARALLAX_BF,
            WorkflowStep.PARALLAX_BF_ACCEPT,
            WorkflowStep.PARALLAX_ALIGNMENT,
            WorkflowStep.PARALLAX_REVIEW,
            WorkflowStep.PARALLAX_ADVANCED,
            WorkflowStep.PARALLAX,
        ):
            state.mark_completed(step)

        state.parameters_updated(WorkflowStep.PARALLAX_BF)

        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX_BF_ACCEPT))
        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX_ALIGNMENT))
        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX_REVIEW))
        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX_ADVANCED))
        self.assertTrue(state.is_stale(WorkflowStep.PARALLAX))


if __name__ == "__main__":
    unittest.main()
