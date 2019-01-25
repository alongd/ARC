#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests for the ARC class
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest

import arc.ts.auto_tst as auto_tst

from rmgpy.molecule.molecule import Molecule

################################################################################


class TestAutoTST(unittest.TestCase):
    """
    Contains unit tests for AutoTST
    """

    def test_generate_reaction_string(self):
        """Test the generate_reaction_string() function"""
        mol1 = Molecule(SMILES=str('C'))
        mol2 = Molecule(SMILES=str('[OH]'))
        mol3 = Molecule(SMILES=str('[CH3]'))
        mol4 = Molecule(SMILES=str('O'))
        mol5 = Molecule(SMILES=str('CC[O]'))
        mol6 = Molecule(SMILES=str('[NH2]'))
        mol7 = Molecule(SMILES=str('[CH]C[O]'))
        mol8 = Molecule(SMILES=str('N'))

        string1 = auto_tst.generate_reaction_string(reactants=[mol1, mol2], products=[mol3, mol4])
        string2 = auto_tst.generate_reaction_string(reactants=[mol5, mol6], products=[mol7, mol8])

        self.assertEqual(string1, 'C+[OH]_[CH3]+O')
        self.assertEqual(string2, 'CC[O]+[NH2]_[CH]C[O]+N')

    def test_run_autotsts(self):
        """Test the run_autotsts() function"""
        mol1 = Molecule(SMILES=str('CCC'))
        mol2 = Molecule(SMILES=str('O[O]'))
        mol3 = Molecule(SMILES=str('[CH2]CC'))
        mol4 = Molecule(SMILES=str('OO'))

        string1 = auto_tst.generate_reaction_string(reactants=[mol1, mol2], products=[mol3, mol4])
        self.assertEqual(string1, 'CCC+[O]O_[CH2]CC+OO')

        xyz_guess = auto_tst.run_autotsts(reaction_string=string1, family=str('H_Abstraction'))
        expected_xyz = """1"""
        self.assertEqual(xyz_guess, expected_xyz)


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
