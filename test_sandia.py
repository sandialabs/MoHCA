"""
Tests approximating unit tests to ensure functionality during refactor/update
"""

from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import sandia

DATA_DIR = Path(r"./test_data")


class TestSandia(unittest.TestCase):
    """
    Test Class for sandia.py
    """

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.input_csv_path_1 = DATA_DIR / 'loc1.csv'
        self.input_csv_path_2 = DATA_DIR / 'loc12.csv'
        self.model_free_result_csv_path_1 = DATA_DIR / 'res1.csv'
        self.model_free_result_csv_path_2 = DATA_DIR / 'res12.csv'
        self.model_result_csv_path_1 = DATA_DIR / 'HC_Results_model_based.csv'

    def get_good_input_data(self, alt=1):
        """
        Get input data, format datetime columns
        """
        if alt == 1:
            input_data = pd.read_csv(self.input_csv_path_1)
        else:
            input_data = pd.read_csv(self.input_csv_path_2)

        input_data['datetime'] = pd.to_datetime(input_data['datetime'])
        input_data.sort_values('datetime', inplace=True)
        return input_data

    def get_df_edit(self):
        """
        create df_edit as defined in hosting_cap
        """
        # load good data
        input_data = self.get_good_input_data()

        good_ndx = sandia.get_consistent_time_index(
            input_data['datetime'].values
            )
        result = sandia.compare_columns(input_data['datetime'], good_ndx)
        self.assertTrue(result)  # verify working creation of good time index

        # dataframe to modify and include other necessary variables
        df_edit = pd.DataFrame()

        # modify data for algorithm
        # sign convention: negative for load, positive for PV injections
        df_edit['datetime'] = pd.to_datetime(input_data['datetime'])
        df_edit['P'] = -1 * input_data['kw_reading']
        df_edit['Q'] = -1 * input_data['kvar_reading']
        df_edit['V'] = input_data['v_reading']

        return df_edit

    def get_inconsistent_index_data(self):
        """
        Drop 6 blocks of indexes totalling 150 data points
        """
        bad_data = self.get_good_input_data()
        indexes_to_break = [
            (30, 50),
            (100, 129),
            (666, 667),
            (3400, 3491),
            (5000, 5004),
            (6000, 6005),
        ]
        for indexes in indexes_to_break:
            bad_data = bad_data.drop(index=range(indexes[0], indexes[1]))

        return bad_data

    def test_hosting_cap(self):
        """
        To ensure calculation is essentially correct
        """
        res = sandia.hosting_cap(
            self.input_csv_path_1,
            self.model_free_result_csv_path_1,
            )
        self.assertEqual(
            int(res['kw_hostable'].values[0]),
            int(7.065456433851)
            )

    def test_hosting_cap_2(self):
        """
        Not super useful, but a test of two customers at once
        """
        res = sandia.hosting_cap(
            self.input_csv_path_2,
            self.model_free_result_csv_path_2,
        )
        self.assertEqual(len(res), 2)

    def test_sanity(self):
        """
        Original test procedure
        """
        sandia.hosting_cap(
            self.input_csv_path_1,
            self.model_free_result_csv_path_1,
            )

        result = sandia.sanity_check(
            self.model_result_csv_path_1,
            self.model_free_result_csv_path_1)

        self.assertEqual(result, True)

    def test_check_consistent_index(self):
        """
        Test of:
        1. datetime conversion (stock functionality of pandas)
        2. consistent time index generation
        3. column comparison function (both True and False results)
        4. correction of time index with nan while maintaining sort order

        Uses single bus test case data
        """
        input_data = self.get_good_input_data()

        good_ndx = sandia.get_consistent_time_index(
            input_data['datetime'].values
            )
        result = sandia.compare_columns(input_data['datetime'], good_ndx)
        self.assertTrue(result)  # verify working creation of good time index

        bad_data = self.get_inconsistent_index_data()

        result = sandia.compare_columns(bad_data['datetime'], good_ndx)
        self.assertFalse(result)  # verify correct column check function

        # fix inconsistent data with nan
        fixed_input = sandia.fix_inconsistent_time_index(bad_data)
        result = sandia.compare_columns(fixed_input['datetime'], good_ndx)
        self.assertTrue(result)  # verify index has been fixed

        # verify correct counts of na
        nan_sum = fixed_input.isna().sum()

        for index in nan_sum.index:
            if index == 'datetime':
                self.assertTrue(nan_sum[index] == 0)
            else:
                self.assertTrue(nan_sum[index] == 150)

    def test_error_block_creation(self):
        """
        Test of error block creation and duration calculations

        Uses single bus test case data
        """
        bad_data = self.get_inconsistent_index_data()
        # will populate missing rows with nan
        bad_data = sandia.fix_inconsistent_time_index(bad_data)
        bad_data_mask = sandia.get_nan_mask(bad_data)
        error_blocks = sandia.get_error_blocks(bad_data_mask)

        self.assertEqual(len(error_blocks), 6)

        # check counting of bad data points in error blocks
        total_errors = 0
        for error_block in error_blocks.values():
            total_errors += error_block['duration']

        self.assertEqual(total_errors, bad_data_mask.sum())

    def test_error_block_handling(self):
        """
        idea is to pass bad data,
        interpolate two erros
        nan all the others
        check the resulting data
        """
        bad_data = self.get_inconsistent_index_data()

        # rename test data to reflect algorithm data
        column_rename = {
            'v_reading': 'V',
            'kw_reading': 'P',
            'kvar_reading': 'Q'
        }
        bad_data.rename(columns=column_rename, inplace=True)

        # will populate missing rows with nan
        bad_data = sandia.fix_inconsistent_time_index(bad_data)
        bad_data.drop(columns=['busname'], inplace=True)

        bad_data_mask = sandia.get_nan_mask(bad_data)
        initial_n_bad = bad_data_mask.sum()

        error_blocks = sandia.get_error_blocks(bad_data_mask)

        fixed_data, fix_report = sandia.fix_error_blocks(
            bad_data,
            error_blocks
        )

        # identify 6 error blocks
        self.assertEqual(len(fix_report), 6)

        # properly sum data points that were interpolated
        fixed_n = fix_report[fix_report['action'] == 'fixed']['duration'].sum()
        self.assertEqual(fixed_n, 5)

        remaining_bad_data_mask = sandia.get_nan_mask(fixed_data)
        self.assertEqual(
            remaining_bad_data_mask.sum(),
            initial_n_bad - fixed_n)

    def test_held_data_operations(self):
        """
        Load good data, convert to format of algorithm and add 3% noise.
        Ensure data with noise has no held values, then inject held values.
        Test held value identification and correction procedures and returns.
        """
        df_edit = self.get_df_edit()

        # add noise to data
        columns_to_check = ['P', 'Q', 'V']
        percent_noise = 0.03
        noise_lims = abs(df_edit[columns_to_check].mean()) * percent_noise
        noise = np.random.normal(
            0,
            noise_lims,
            df_edit[columns_to_check].shape)
        noise_df = df_edit[columns_to_check] + noise

        held_mask_1 = sandia.get_held_value_mask(noise_df)
        self.assertEqual(held_mask_1.sum(), 0)

        # make bad data
        held_df = noise_df.copy()
        indexes_to_break = [
            (30, 50),  # 20
            (100, 120),  # 20
            (666, 667),  # 1
            (3400, 3403),  # 3
            (5000, 5004),  # 4
            (6000, 6005),  # 5
        ]

        for indexes in indexes_to_break:
            col = np.random.choice(columns_to_check)
            held_df.loc[indexes[0]:indexes[1]-1, col] = \
                np.round(held_df[col][indexes[0]], 0)

        # identify held data
        held_mask_2 = sandia.get_held_value_mask(held_df)
        self.assertEqual(held_mask_2.sum(), 49)  # held locations sum of >= 4

        # test alternative threshold input
        held_mask_3 = sandia.get_held_value_mask(
            held_df,
            held_n_threshold=10)
        self.assertEqual(held_mask_3.sum(), 40)  # held locations sum of >= 10

        # identify error blocks of held data
        error_blocks = sandia.get_error_blocks(held_mask_2)
        self.assertEqual(len(error_blocks), 4)

        # fix error blocks
        fixed_data, error_df = sandia.fix_error_blocks(
            held_df,
            error_blocks,
            interpolation_threshold=6
        )

        # ensure replacement of nans during fix
        nan_mask = sandia.get_nan_mask(fixed_data)
        self.assertEqual(nan_mask.sum(), 40)

        # ensure only 2 blocks fixed
        self.assertEqual(
            error_df['action'].value_counts()['fixed'], 2
        )

        # ensure fixing of helds
        post_fix_held_mask = sandia.get_held_value_mask(fixed_data)
        self.assertEqual(post_fix_held_mask.sum(), 0)

    def test_voltage_mask(self):
        """
        testing of vpu conversion and following voltage range masking
        """
        df_edit = self.get_df_edit()
        df_edit['v_base'] = sandia.get_v_base(df_edit)
        df_edit['vpu'] = sandia.get_vpu(df_edit)
        mask = sandia.get_bad_voltage_mask(df_edit)
        self.assertEqual(mask.sum(), 0)

        # inssert bad data
        df_edit.loc[20, 'vpu'] = 1.4
        df_edit.loc[60, 'vpu'] = 1.2
        df_edit.loc[300, 'vpu'] = 0.4
        df_edit.loc[400, 'vpu'] = 0.8
        df_edit.loc[500, 'vpu'] = 1
        df_edit.loc[600, 'vpu'] = 0.1

        # ensure dat data dectected
        mask = sandia.get_bad_voltage_mask(df_edit)
        self.assertEqual(mask.sum(), 5)

    def test_has_input_q_check(self):
        """
        test missing q
        """
        input_data = self.get_good_input_data()

        passing_test = sandia.get_has_input_q(input_data)
        self.assertTrue(passing_test)

        input_data['kvar_reading'] = np.nan
        failing_test = sandia.get_has_input_q(input_data)

        self.assertFalse(failing_test)


if __name__ == '__main__':
    unittest.main()
