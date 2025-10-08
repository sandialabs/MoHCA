"""
Tests approximating unit tests to ensure functionality during refactor/update
"""

from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import sandia_TCHC

import transformer_customer_mapping_isu as tcm_isu

PARENT_DIR = Path("__file__").parent.absolute()
DATA_DIR = Path(PARENT_DIR, "mohca_cl", "test_data", "ST_Example_Files")

class TestSandiaTCHC(unittest.TestCase):
    """
    Test Class for sandia_TCHC.py
    """

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # inputs
        self.hv_input_fp = DATA_DIR / 'ST_input_meter_data.csv'  # higher v
        self.isu_input_fp = DATA_DIR / 'ST_Input_ISU.csv'  # temp...
        # optional inputs
        self.optional_bus_coords_1 = DATA_DIR / "ST_optional_bus_coords.csv"

        # expected output
        self.expected_grouping_path = DATA_DIR / 'ST_transformer_customer_mapping_expected.csv'
        self.expected_tchc_path = DATA_DIR / 'ST_tchc_results_expected_default_settings.csv'
        self.expected_tchc_path_actual_xf = DATA_DIR / 'ST_tchc_results_actual_xf_lookup.csv'

        # test outputs
        self.grouping_1_output = DATA_DIR / 'output' / 'ST_transformer_customer_mapping.csv'
        self.tchc_output_path = DATA_DIR / 'output' / 'ST_tchc_results.csv'

    def test_tcm_01_pass(self):
        """
        Test to verify operation of transformer customer mapping (grouping)
        """
        input_meter_data_fp = self.isu_input_fp  # for passing test
        grouping_output_fp = self.grouping_1_output

        bus_coords_fp = self.optional_bus_coords_1
        minimum_xfmr_n = 12  # required minimum value
        xfmr_n_is_exact = True

        # passing test
        grouping_results = tcm_isu.get_groupings(
            input_meter_data_fp,
            grouping_output_fp,
            minimum_xfmr_n,
            xfmr_n_is_exact,
            bus_coords_fp
            )
        grouping_results_expected = pd.read_csv(self.expected_grouping_path)
        match = sandia_TCHC.check_results(
            grouping_results,
            grouping_results_expected)
        self.assertEqual(True, match)

    def test_tcm_01_est_pass(self):
        """
        Test to verify operation of transformer customer mapping (grouping)
        using NO optional inputs (xfmr num or bus coords)
        """
        input_meter_data_fp = self.isu_input_fp  # for passing test
        grouping_output_fp = self.grouping_1_output

        # passing test
        grouping_results = tcm_isu.get_groupings(
            input_meter_data_fp,
            grouping_output_fp,
            )
        # expected output is same as exact, but no coords
        grouping_results_expected = pd.read_csv(self.expected_grouping_path)
        no_coord_val = 0.0  # NOTE: used to fill coordinates if none.
        grouping_results_expected['X'] = no_coord_val
        grouping_results_expected['Y'] = no_coord_val

        match = sandia_TCHC.check_results(
            grouping_results,
            grouping_results_expected)
        self.assertEqual(True, match)

    def test_tcm_01_est_pass_with_coords(self):
        """
        Test to verify operation of transformer customer mapping (grouping)
        optional input is only bus corrdinates
        """
        input_meter_data_fp = self.isu_input_fp  # for passing test
        grouping_output_fp = self.grouping_1_output

        bus_coords_fp = self.optional_bus_coords_1
        minimum_xfmr_n = None  # required minimum value
        xfmr_n_is_exact = False

        # passing test
        grouping_results = tcm_isu.get_groupings(
            input_meter_data_fp,
            grouping_output_fp,
            minimum_xfmr_n,
            xfmr_n_is_exact,
            bus_coords_fp
            )
        # expected output is same as exact
        grouping_results_expected = pd.read_csv(self.expected_grouping_path)

        match = sandia_TCHC.check_results(
            grouping_results,
            grouping_results_expected)
        self.assertEqual(True, match)

    def test_tcm_01_fail(self):
        """
        Test to verify operation of transformer customer mapping (grouping)
        """
        failing_input_fp = self.hv_input_fp  # for failing test
        grouping_output_fp = self.grouping_1_output
        bus_coords_fp = self.optional_bus_coords_1
        minimum_xfmr_n = 12  # required minimum value

        # failing test
        grouping_results = tcm_isu.get_groupings(
            failing_input_fp,
            grouping_output_fp,
            minimum_xfmr_n,  # not optional
            bus_coords_fp
            )
        grouping_results_expected = pd.read_csv(self.expected_grouping_path)
        match = sandia_TCHC.check_results(
            grouping_results,
            grouping_results_expected)
        self.assertEqual(False, match)

    def test_tcm_01_no_bus_coords(self):
        """
        Test to verify operation of transformer customer mapping (grouping)
        """
        input_meter_data_fp = self.isu_input_fp  # for passing test
        grouping_output_fp = self.grouping_1_output
        minimum_xfmr_n = 12  # required minimum value
        xfmr_n_is_exact = True

        grouping_results = tcm_isu.get_groupings(
            input_meter_data_fp,
            grouping_output_fp,
            minimum_xfmr_n,
            xfmr_n_is_exact
            )
        grouping_results_expected = pd.read_csv(self.expected_grouping_path)

        # remove coordinates from results.
        no_coord_val = 0.0  # NOTE: used to fill coordinates if none.
        grouping_results_expected['X'] = no_coord_val
        grouping_results_expected['Y'] = no_coord_val

        match = sandia_TCHC.check_results(
            grouping_results,
            grouping_results_expected)
        self.assertEqual(True, match)

    def test_sandia_tchc(self):
        """
        Original test procedure... takes ~9 sec

        Currently setup to use the 'higher voltage' inputs...
        """
        tcm_grouping = pd.read_csv(self.expected_grouping_path)

        res = sandia_TCHC.hosting_cap_tchc(
            self.hv_input_fp,
            self.tchc_output_path,
            tcm_grouping
            )

        ref_res = pd.read_csv(self.expected_tchc_path)
        match = sandia_TCHC.check_results(ref_res, res)

        self.assertEqual(match, True)

    def test_sandia_tchc_xf_input(self):
        """
        acceptance of xf_lookup test
        doesn't alter expected outputs
        """
        tcm_grouping = pd.read_csv(self.expected_grouping_path)

        xf_lookup = pd.DataFrame(columns=['kVA', 'R_ohms_LV', 'X_ohms_LV'])
        xf_lookup['kVA'] = [50]
        xf_lookup['R_ohms_LV'] = [0.0135936]
        xf_lookup['X_ohms_LV'] = [0.0165888]

        res = sandia_TCHC.hosting_cap_tchc(
            self.hv_input_fp,
            self.tchc_output_path,
            tcm_grouping,
            xf_lookup=xf_lookup
            )

        ref_res = pd.read_csv(self.expected_tchc_path_actual_xf)
        match = sandia_TCHC.check_results(ref_res, res)

        self.assertEqual(match, True)

    def test_sandia_tchc_xf_input_fail(self):
        """
        To test if outputs are actually different with altered xf lookup
        """
        tcm_grouping = pd.read_csv(self.expected_grouping_path)

        xf_lookup = pd.DataFrame(columns=['kVA', 'R_ohms_LV', 'X_ohms_LV'])
        xf_lookup['kVA'] = [25]
        xf_lookup['R_ohms_LV'] = [3]
        xf_lookup['X_ohms_LV'] = [0.2]

        res = sandia_TCHC.hosting_cap_tchc(
            self.hv_input_fp,
            self.tchc_output_path,
            tcm_grouping,
            xf_lookup=xf_lookup
            )

        ref_res = pd.read_csv(self.expected_tchc_path_actual_xf)
        match = sandia_TCHC.check_results(ref_res, res)

        self.assertEqual(match, False)


if __name__ == '__main__':
    unittest.main()
