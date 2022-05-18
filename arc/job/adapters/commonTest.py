#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.common module
"""

import os
import unittest

import arc.job.adapters.common as common
from arc.common import ARC_PATH
from arc.job.adapters.gaussian import GaussianAdapter
from arc.job.adapters.molpro import MolproAdapter
from arc.level import Level
from arc.species import ARCSpecies


class TestJobCommon(unittest.TestCase):
    """
    Contains unit tests for the job.adapters.common module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = GaussianAdapter(execution_type='incore',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_2 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    level=Level(method='wb97xd', basis='def2tzvp'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_3 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    level=Level(method='wb97xd', basis='def2tzvp'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1, number_of_radicals=2)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_4 = MolproAdapter(execution_type='incore',
                                  job_type='irc',
                                  level=Level(method='ccsd(t)', basis='cc-pvtz'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                                  testing=True,
                                  )
        cls.job_5 = GaussianAdapter(execution_type='incore',
                                    job_type='irc',
                                    level=Level(method='b3lyp', basis='def2svp'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_6 = MolproAdapter(execution_type='incore',
                                  job_type='scan',
                                  level=Level(method='ccsd(t)', basis='cc-pvtz'),
                                  project='test',
                                  project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_MolproAdapter'),
                                  species=[ARCSpecies(label='ethane', smiles='CC')],
                                  testing=True,
                                  )

    def test_is_restricted(self):
        """Test the is_restricted() function"""
        self.assertTrue(common.is_restricted(self.job_1))
        self.assertFalse(common.is_restricted(self.job_2))
        self.assertFalse(common.is_restricted(self.job_3))

    def test_check_argument_consistency(self):
        """Test the check_argument_consistency() function"""
        common.check_argument_consistency(self.job_1)
        common.check_argument_consistency(self.job_2)
        with self.assertRaises(NotImplementedError):
            common.check_argument_consistency(self.job_4)
        with self.assertRaises(ValueError):
            common.check_argument_consistency(self.job_5)
        self.job_6.species[0].determine_rotors()
        with self.assertRaises(NotImplementedError):
            common.check_argument_consistency(self.job_6)
        self.job_2.scan_res = 55.6
        with self.assertRaises(ValueError):
            common.check_argument_consistency(self.job_2)
        with self.assertRaises(ValueError):
            common.check_argument_consistency(self.job_3)

    def test_update_input_dict_with_args(self):
        """Test the update_input_dict_with_args() function"""
        input_dict = common.update_input_dict_with_args(args={}, input_dict={})
        self.assertEqual(input_dict, dict())

        input_dict = common.update_input_dict_with_args(args={'block': """a\nb"""}, input_dict={})
        self.assertEqual(input_dict, {'block': """\n\na\nb"""})

        input_dict = common.update_input_dict_with_args(args={'block': """a\nb"""}, input_dict={'block': """x\ny\n"""})
        self.assertEqual(input_dict, {'block': """x\ny\na\nb\n"""})

        input_dict = common.update_input_dict_with_args(args={'keyword': {'scan_trsh': 'keyword 1'}}, input_dict={})
        self.assertEqual(input_dict, {'scan_trsh': 'keyword 1'})

        input_dict = common.update_input_dict_with_args(args={'keyword': {'opt': 'keyword 2'}}, input_dict={})
        self.assertEqual(input_dict, {'keywords': 'keyword 2'})

    def test_set_job_args(self):
        """Test the set_job_args() function"""
        args = common.set_job_args(args=None, level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword': dict(), 'block': dict(), 'trsh': dict()})

        args = common.set_job_args(args=dict(), level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword': dict(), 'block': dict(), 'trsh': dict()})

        args = common.set_job_args(args={'keyword': 'k1'}, level=Level(repr='CBS-QB3'), job_name='j1')
        self.assertEqual(args, {'keyword':'k1', 'block': dict(), 'trsh': dict()})

    def test_which(self):
        """Test the which() function"""
        ans = common.which(command='python', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='python', return_bool=False, raise_error=False)
        self.assertIn('arc_env/bin/python', ans)

        ans = common.which(command='ls', return_bool=True, raise_error=False)
        self.assertTrue(ans)

        ans = common.which(command='fake_command_1', return_bool=True, raise_error=False)
        self.assertFalse(ans)

        ans = common.which(command=['python'], return_bool=False, raise_error=False)
        self.assertIn('bin/python', ans)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
