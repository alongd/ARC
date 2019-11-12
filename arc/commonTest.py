#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's common module
"""

import copy
import os
import time
import unittest

import arc.common as common
from arc.exceptions import InputError, SettingsError
from arc.settings import arc_path, servers


class TestCommon(unittest.TestCase):
    """
    Contains unit tests for ARC's common module
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.default_job_types = {'conformers': True,
                                 'opt': True,
                                 'fine': True,
                                 'freq': True,
                                 'sp': True,
                                 'rotors': True,
                                 'orbitals': False,
                                 'onedmin': False,
                                 'bde': False,
                                 }

    def test_read_yaml_file(self):
        """Test the read_yaml_file() function"""
        restart_path = os.path.join(arc_path, 'arc', 'testing', 'restart', 'restart(H,H2O2,N2H3,CH3CO2).yml')
        input_dict = common.read_yaml_file(restart_path)
        self.assertIsInstance(input_dict, dict)
        self.assertTrue('reactions' in input_dict)
        self.assertTrue('freq_level' in input_dict)
        self.assertTrue('use_bac' in input_dict)
        self.assertTrue('ts_guess_level' in input_dict)
        self.assertTrue('running_jobs' in input_dict)

        with self.assertRaises(InputError):
            common.read_yaml_file('nopath')

    def test_get_git_commit(self):
        """Test the get_git_commit() function"""
        git_commit = common.get_git_commit()
        # output format: ['fafdb957049917ede565cebc58b29899f597fb5a', 'Fri Mar 29 11:09:50 2019 -0400']
        self.assertEqual(len(git_commit[0]), 40)
        self.assertEqual(len(git_commit[1].split()), 6)
        self.assertIsInstance(git_commit[0], str)
        self.assertIsInstance(git_commit[1], str)

    def test_get_git_branch(self):
        """Test the get_git_branch() function"""
        git_branch = common.get_git_branch()
        self.assertIsInstance(git_branch, str)

    def test_time_lapse(self):
        """Test the time_lapse() function"""
        t0 = time.time()
        time.sleep(2)
        lap = common.time_lapse(t0)
        self.assertEqual(lap, '00:00:02')

    def test_check_ess_settings(self):
        """Test the check_ess_settings function"""
        server_names = list(servers.keys())
        ess_settings1 = {'gaussian': [server_names[0]], 'molpro': [server_names[1], server_names[0]],
                         'qchem': [server_names[0]]}
        ess_settings2 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0]}
        ess_settings3 = {'gaussian': server_names[0], 'molpro': [server_names[1], server_names[0]],
                         'qchem': server_names[0]}
        ess_settings4 = {'gaussian': server_names[0], 'molpro': server_names[1], 'qchem': server_names[0]}
        ess_settings5 = {'gaussian': 'local', 'molpro': server_names[1], 'qchem': server_names[0]}

        ess_settings1 = common.check_ess_settings(ess_settings1)
        ess_settings2 = common.check_ess_settings(ess_settings2)
        ess_settings3 = common.check_ess_settings(ess_settings3)
        ess_settings4 = common.check_ess_settings(ess_settings4)
        ess_settings5 = common.check_ess_settings(ess_settings5)

        ess_list = [ess_settings1, ess_settings2, ess_settings3, ess_settings4, ess_settings5]

        for ess in ess_list:
            for soft, server_list in ess.items():
                self.assertTrue(soft in ['gaussian', 'molpro', 'qchem'])
                self.assertIsInstance(server_list, list)

        with self.assertRaises(SettingsError):
            ess_settings6 = {'nosoft': ['server1']}
            common.check_ess_settings(ess_settings6)
        with self.assertRaises(SettingsError):
            ess_settings7 = {'gaussian': ['noserver']}
            common.check_ess_settings(ess_settings7)

    def test_min_list(self):
        """Test the min_list() function"""
        lst = []
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [None, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, None)

        lst = [0]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, 0)

        lst = [-8]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -80]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -80)

        lst = [-8, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, -8, -8, -8]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -8)

        lst = [-8, None, None, 100, -79, None]
        min_lst = common.min_list(lst)
        self.assertEqual(min_lst, -79)

    def test_key_by_val(self):
        d = {1: 5, 2: 8}
        self.assertEqual(common.key_by_val(d, 8), 2)

        d = {1: 5, 2: 8, 5: 8}
        self.assertIn(common.key_by_val(d, 8), [2, 5])

        d = {1: 5, 2: None, 3: 9}
        self.assertEqual(common.key_by_val(d, 9), 3)

        d = {1: 5, 2: 'X', 3: 9}
        self.assertEqual(common.key_by_val(d, 9), 3)

        with self.assertRaises(ValueError):
            common.key_by_val(d, 10)

    def test_initialize_job_with_given_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {'conformers': False, 'opt': True, 'fine': True, 'freq': True, 'sp': False, 'rotors': False}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_empty_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_bde_job_type(self):
        """Test the initialize_job_types() function"""
        job_types = {'bde': True}
        job_types_expected = copy.deepcopy(self.default_job_types)
        job_types_expected.update(job_types)
        job_types_initialized = common.initialize_job_types(job_types)
        self.assertEqual(job_types_expected, job_types_initialized)

    def test_initialize_job_with_specific_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'freq'
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected[specific_job_type] = True
        specific_job_type_initialized = common.initialize_job_types({}, specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_conflict_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'sp'
        conflict_job_type = {'bde': True}
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected[specific_job_type] = True
        specific_job_type_initialized = common.initialize_job_types(conflict_job_type,
                                                                    specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_specific_bde_job_type(self):
        """Test the initialize_job_types() function"""
        specific_job_type = 'bde'
        bde_default = {'opt': True, 'fine': True, 'freq': True, 'sp': True, 'bde': True}
        specific_job_type_expected = {job_type: False for job_type in self.default_job_types.keys()}
        specific_job_type_expected.update(bde_default)
        specific_job_type_initialized = common.initialize_job_types({}, specific_job_type=specific_job_type)
        self.assertEqual(specific_job_type_expected, specific_job_type_initialized)

    def test_initialize_job_with_not_supported_job_type(self):
        """Test the initialize_job_types() function"""
        with self.assertRaises(InputError):
            job_types = {'fake_job': True}
            common.initialize_job_types(job_types)
        with self.assertRaises(InputError):
            specific_job_type = 'fake_job_type'
            common.initialize_job_types({}, specific_job_type=specific_job_type)

    def test_determine_ess(self):
        """Test the determine_ess function"""
        gaussian = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        qchem = os.path.join(arc_path, 'arc', 'testing', 'C2H6_freq_QChem.out')
        molpro = os.path.join(arc_path, 'arc', 'testing', 'CH2O_freq_molpro.out')

        self.assertEqual(common.determine_ess(gaussian), 'gaussian')
        self.assertEqual(common.determine_ess(qchem), 'qchem')
        self.assertEqual(common.determine_ess(molpro), 'molpro')

    def test_sort_two_lists_by_the_first(self):
        """Test the sort_two_lists_by_the_first function"""
        list1 = [5, 2, 8, 1, 0]
        list2 = ['D', 'C', 'E', 'B', 'A']
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [0, 1, 2, 5, 8])
        self.assertEqual(list2, ['A', 'B', 'C', 'D', 'E'])

        list1 = [-402175.42413054925, -402175.42413054925, -402175.42413054925]
        list2 = [
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.289877, -1.027889, -0.472673), (-0.705088, 0.018405, 0.167196),
                        (0.681617, 0.004981, -0.157809), (-1.21258, 0.938155, 0.439627),
                        (-1.264105, -0.911112, 0.141353), (1.210279, 0.97746, -0.117693))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.438259, -0.905929, 0.233373), (-0.722557, -0.021223, 0.05388),
                        (0.695838, 0.042156, -0.059689), (-1.184869, -0.935184, 0.411808),
                        (-1.348775, 0.824904, -0.209994), (1.122104, 0.995276, -0.429377))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((-1.097258, 1.288817, 0.280596), (0.66579, -0.163586, -0.235383),
                        (-0.646392, 0.135474, 0.230984), (1.308294, 0.646397, -0.564452),
                        (1.03492, -1.183579, -0.268027), (-1.265354, -0.723524, 0.556282))}]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [-402175.42413054925, -402175.42413054925, -402175.42413054925])
        self.assertEqual(list2, [
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.289877, -1.027889, -0.472673), (-0.705088, 0.018405, 0.167196),
                        (0.681617, 0.004981, -0.157809), (-1.21258, 0.938155, 0.439627),
                        (-1.264105, -0.911112, 0.141353), (1.210279, 0.97746, -0.117693))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((1.438259, -0.905929, 0.233373), (-0.722557, -0.021223, 0.05388),
                        (0.695838, 0.042156, -0.059689), (-1.184869, -0.935184, 0.411808),
                        (-1.348775, 0.824904, -0.209994), (1.122104, 0.995276, -0.429377))},
            {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
             'coords': ((-1.097258, 1.288817, 0.280596), (0.66579, -0.163586, -0.235383),
                        (-0.646392, 0.135474, 0.230984), (1.308294, 0.646397, -0.564452),
                        (1.03492, -1.183579, -0.268027), (-1.265354, -0.723524, 0.556282))}])

        list1 = [-293712.44825524034, -293712.4484442763, -293719.9392868027]
        list2 = [
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((0.626959, 0.049055, 0.398851), (-0.627579, -0.002911, -0.40088),
                        (1.135191, 0.833831, -0.015525), (1.156572, -0.755054, 0.054181),
                        (-1.135811, -0.787687, 0.013496), (-1.157192, 0.801198, -0.056209))},
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((0.650715, 0.034647, 0.359727), (-0.650716, -0.034614, -0.359727),
                        (1.181164, -0.741554, -0.042786), (1.107, 0.847165, -0.061582),
                        (-1.106969, -0.847181, 0.061521), (-1.181193, 0.741537, 0.042846))},
            {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (14, 14, 1, 1, 1, 1),
             'coords': ((-0.679412, 0.224819, -0.236977), (0.680753, -0.233287, -0.22462),
                        (-0.974999, 0.570328, 0.676502), (-1.260937, -0.580211, -0.451942),
                        (1.263604, 0.56338, -0.465582), (0.970991, -0.54503, 0.702618))}]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [-293719.9392868027, -293712.4484442763, -293712.44825524034])
        self.assertEqual(list2, [
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((-0.679412, 0.224819, -0.236977), (0.680753, -0.233287, -0.22462),
                        (-0.974999, 0.570328, 0.676502), (-1.260937, -0.580211, -0.451942),
                        (1.263604, 0.56338, -0.465582), (0.970991, -0.54503, 0.702618))},
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((0.650715, 0.034647, 0.359727), (-0.650716, -0.034614, -0.359727),
                        (1.181164, -0.741554, -0.042786), (1.107, 0.847165, -0.061582),
                        (-1.106969, -0.847181, 0.061521), (-1.181193, 0.741537, 0.042846))},
            {'isotopes': (14, 14, 1, 1, 1, 1), 'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
             'coords': ((0.626959, 0.049055, 0.398851), (-0.627579, -0.002911, -0.40088),
                        (1.135191, 0.833831, -0.015525), (1.156572, -0.755054, 0.054181),
                        (-1.135811, -0.787687, 0.013496), (-1.157192, 0.801198, -0.056209))}])

        list1 = [5, None, 1]
        list2 = [1, 2, 3]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [1, 5])
        self.assertEqual(list2, [3, 1])

        list1 = [None, None, None]
        list2 = [1, 2, 3]
        list1, list2 = common.sort_two_lists_by_the_first(list1, list2)
        self.assertEqual(list1, [])
        self.assertEqual(list2, [])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
