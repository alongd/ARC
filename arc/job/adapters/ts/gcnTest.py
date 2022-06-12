#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.gcn module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.ts.gcn_ts import GCNAdapter, HAS_GCN
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.species.species import ARCSpecies


class TestGCNAdapter(unittest.TestCase):
    """
    Contains unit tests for the GCNAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)
        cls.output_dir = os.path.join(ARC_PATH, 'arc', 'testing', 'GCN')
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)

    def test_gcn_ring_cleavage(self):
        """
        Test that ARC can call GCN to make TS guesses for ring cleavage reactions.
        """
        reactant_xyz = """C  -1.3087    0.0068    0.0318
                          C   0.1715   -0.0344    0.0210
                          N   0.9054   -0.9001    0.6395
                          O   2.1683   -0.5483    0.3437
                          N   2.1499    0.5449   -0.4631
                          N   0.9613    0.8655   -0.6660
                          H  -1.6558    0.9505    0.4530
                          H  -1.6934   -0.0680   -0.9854
                          H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='reactant', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)

        product_xyz = """C  -1.0108   -0.0114   -0.0610
                         C   0.4780    0.0191    0.0139
                         N   1.2974   -0.9930    0.4693
                         O   0.6928   -1.9845    0.8337
                         N   1.7456    1.9701   -0.6976
                         N   1.1642    1.0763   -0.3716
                         H  -1.4020    0.9134   -0.4821
                         H  -1.3327   -0.8499   -0.6803
                         H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='product', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)

        rxn = ARCReaction(label='reactant <=> product', ts_label='TS0', r_species=[reactant], p_species=[product])

        gcn = GCNAdapter(job_type='tsg',
                         reactions=[rxn],
                         testing=True,
                         project='test',
                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN1'),
                         dihedral_increment=2,
                         )
        gcn.local_path = self.output_dir
        gcn.execute_incore()

        self.assertEqual(rxn.ts_species.multiplicity, 1)
        self.assertTrue(len(rxn.ts_species.ts_guesses))
        self.assertEqual(rxn.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'))
        self.assertEqual(rxn.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'))
        self.assertEqual(len(rxn.ts_species.ts_guesses[1].initial_xyz['coords']), 9)
        for tsg in rxn.ts_species.ts_guesses:
            self.assertTrue(tsg.success)
        self.assertLess(rxn.ts_species.ts_guesses[0].execution_time.seconds, 300)  # 0:00:01.985336

    def test_gcn_intra_h_migration(self):
        """
        Test that ARC can call GCN to make TS guesses for intra-H migration reactions.
        """
        reactant_xyz = """C      -0.45684508   -0.05786787   -0.03035793
O       0.82884092   -0.36979664    0.26059161
H      -0.64552049    0.90546331   -0.47022606
H      -1.17752551   -0.82040960    0.21231033
H       0.92128715   -1.25559200    0.65522542"""
        reactant = ARCSpecies(label='reactant', smiles='[CH2]O', xyz=reactant_xyz)

        product_xyz = """C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440"""
        product = ARCSpecies(label='product', smiles='C[O]', xyz=product_xyz)

        rxn = ARCReaction(label='reactant <=> product', ts_label='TS0', r_species=[reactant], p_species=[product])

        gcn = GCNAdapter(job_type='tsg',
                         reactions=[rxn],
                         testing=True,
                         project='test',
                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN2'),
                         dihedral_increment=5.9,
                         )
        gcn.local_path = self.output_dir
        gcn.execute_incore()

        self.assertEqual(rxn.ts_species.multiplicity, 2)
        self.assertTrue(len(rxn.ts_species.ts_guesses))
        success = False
        for tsg in rxn.ts_species.ts_guesses:
            if tsg.success:
                self.assertEqual(tsg.initial_xyz['symbols'], ('C', 'O', 'H', 'H', 'H'))
                self.assertEqual(len(tsg.initial_xyz['coords']), 5)
                self.assertLess(tsg.execution_time.seconds, 300)  # 0:00:01.985336
                success = True
        self.assertTrue(success)

    def test_gcn_keto_enol(self):
        """
        Test that ARC can call GCN to make TS guesses for keto-enol reactions.
        """
        reactant_xyz = """C      -0.80601307   -0.11773769    0.32792128
C       0.23096883    0.47536513   -0.26437348
O       1.44620485   -0.11266560   -0.46339257
H      -1.74308628    0.41660480    0.45016601
H      -0.75733964   -1.13345488    0.70278513
H       0.21145717    1.48838416   -0.64841675
H       1.41780836   -1.01649567   -0.10468897"""
        reactant = ARCSpecies(label='reactant', smiles='C=CO', xyz=reactant_xyz)

        product_xyz = """C      -0.64851652   -0.03628781   -0.04007233
C       0.84413281    0.04088405    0.05352862
O       1.47323666   -0.23917853    1.06850992
H      -1.06033881    0.94648764   -0.28238370
H      -0.92134271   -0.74783968   -0.82281679
H      -1.04996634   -0.37234114    0.91874740
H       1.36260637    0.37153887   -0.86221771"""
        product = ARCSpecies(label='product', smiles='CC=O', xyz=product_xyz)

        rxn = ARCReaction(label='reactant <=> product', ts_label='TS0', r_species=[reactant], p_species=[product])

        gcn = GCNAdapter(job_type='tsg',
                         reactions=[rxn],
                         testing=True,
                         project='test',
                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN3'),
                         dihedral_increment=5,
                         )
        gcn.local_path = self.output_dir
        gcn.execute_incore()

        self.assertEqual(rxn.ts_species.multiplicity, 1)
        self.assertTrue(len(rxn.ts_species.ts_guesses))
        success = False
        for tsg in rxn.ts_species.ts_guesses:
            if tsg.success:
                self.assertEqual(tsg.initial_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H'))
                self.assertEqual(len(tsg.initial_xyz['coords']), 7)
                self.assertLess(tsg.execution_time.seconds, 300)  # 0:00:01.985336
                success = True
        self.assertTrue(success)

    def test_gcn_no2_ono(self):
        """
        Test that ARC can call GCN to make TS guesses for NO2-ONO reactions.
        """
        reactant_xyz = """C      -1.11302684    0.21797155   -0.07991772
C       0.17440392   -0.27602923    0.54557406
O       1.87811745    1.06430779   -0.32573670
O       1.55410309   -0.92039169   -1.22630419
N       1.29630973   -0.02410965   -0.41326375
H      -1.32555036   -0.30290527   -1.01986488
H      -1.95349237    0.04674238    0.59983262
H      -1.06538939    1.29067141   -0.29677143
H       0.40845145    0.25545042    1.47285390
H       0.14607362   -1.35170585    0.74359893"""
        reactant = ARCSpecies(label='reactant', smiles='CC[N+]([O-])=O', xyz=reactant_xyz)

        product_xyz = """C      -1.36894499    0.07118059   -0.24801399
C      -0.01369535    0.17184136    0.42591278
O      -0.03967083   -0.62462610    1.60609048
N       1.23538512   -0.53558048    2.24863846
O       1.25629155   -1.21389295    3.27993827
H      -2.16063255    0.41812452    0.42429392
H      -1.39509985    0.66980796   -1.16284741
H      -1.59800183   -0.96960842   -0.49986392
H       0.19191326    1.21800574    0.68271847
H       0.76371340   -0.19234475   -0.25650067"""
        product = ARCSpecies(label='product', smiles='CCON=O', xyz=product_xyz)

        rxn = ARCReaction(label='reactant <=> product', ts_label='TS0', r_species=[reactant], p_species=[product])

        gcn = GCNAdapter(job_type='tsg',
                         reactions=[rxn],
                         testing=True,
                         project='test',
                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN4'),
                         dihedral_increment=5,
                         )
        gcn.local_path = self.output_dir
        gcn.execute_incore()

        self.assertEqual(rxn.ts_species.multiplicity, 1)
        self.assertTrue(len(rxn.ts_species.ts_guesses))
        success = False
        for tsg in rxn.ts_species.ts_guesses:
            if tsg.success:
                self.assertEqual(tsg.initial_xyz['symbols'], ('C', 'C', 'O', 'O', 'N', 'H', 'H', 'H', 'H', 'H'))
                self.assertEqual(len(tsg.initial_xyz['coords']), 10)
                self.assertLess(tsg.execution_time.seconds, 300)  # 0:00:01.985336
                success = True
        self.assertTrue(success)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'GCN'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN1'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN2'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN3'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_GCN4'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
