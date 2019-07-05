#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module for (non-TS) species conformer generation

Note that variables that contain atom indices such as torsions and tops are 1-indexed,
while atoms in Molecules are 0-indexed

Todo:
    * determine the number of combinations from the wells
    * have a max number of confs as constant and as user input to override it
    * determine by FF scans which wells to consider for the combinations
    * decide on a template for saving them to a file (YAML)
    * Consider boat-chair conformers
    * finally, consider h-bonds
    ** Does it take the scan energy into account when generating combinations??
    ** The secretary problem - incorporate for stochastic searching
    ** Chirality of lowest conf (start by FF real CJ and my CJ; then pass chiral SMILES and get a CJ conf)
    ** What's the confirmed bottleneck?
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from itertools import product
import time
import numpy as np
from heapq import nsmallest

from rdkit import Chem
from rdkit.Chem.rdchem import EditableMol as RDMol
from rdkit.Chem.rdmolops import AssignStereochemistry
import openbabel as ob
import pybel as pyb

from rmgpy.molecule.molecule import Molecule
from rmgpy.molecule.converter import toOBMol
import rmgpy.molecule.group as gr

from arc.arc_exceptions import ConformerError
from arc.species import converter
from arc.common import logger
import arc.plotter

##################################################################


# conformers is a list of dictionaries, each with the following keys:
# {'xyz': <str>,
#  'index': <int>,
#  'FF energy': <float>,
#  'source': <str>,
#  'torsion_dihedrals': {<torsion tuple 0>: angle 0,
#                        <torsion tuple 1>: angle 1,
#  }

# Module workflow:
# generate_conformers
#     generate_force_field_conformers
#         get_force_field_energies, rdkit_force_field or mix_rdkit_and_openbabel_force_field, determine_dihedrals
#     deduce_new_conformers
#         get_torsion_angles, determine_torsion_symmetry, determine_torsion_sampling_points,
#         change_dihedrals_and_force_field_it
#     get_lowest_confs

# The number of conformers to generate per range of heavy atoms in the molecule
CONFS_VS_HEAVY_ATOMS = {(0, 1): 1,
                        (2, 2): 5,
                        (3, 9): 100,
                        (10, 29): 500,
                        (30, 59): 1000,
                        (60, 99): 5000,
                        (100, 'inf'): 10000,
                        }

# The number of conformers to generate per range of potential torsions in the molecule
CONFS_VS_TORSIONS = {(0, 1): 10,
                     (2, 5): 50,
                     (5, 24): 500,
                     (25, 49): 5000,
                     (50, 'inf'): 10000,
                     }

# The resolution (in degrees) for scanning smeared wells
SMEARED_SCAN_RESOLUTIONS = 30.0

# The maximal number of conformers to consider when generating combinations of wells
MAX_CONFS_NUM = 1e5

# The number of conformers to return. Will be iteratively checked for consistency. The rest will be written to a file.
CONFS_TO_RETURN = 10

# An energy threshold (in kJ/mol) above which wells in a torsion will not be considered (rel. to the most stable well)
DE_THRESHOLD = 5.

# The gap (in degrees) that defines different wells
WELL_GAP = 20

# The maximum number of times to iteratively search for the lowest conformer
MAX_COMBINATION_ITERATIONS = 25

# A threshold below which all combinations will be generated. Above it just samples of the entire search space.
COMBINATION_THRESHOLD = 1000


def generate_conformers(mol_list, label, xyzs=None, torsions=None, tops=None, charge=0, multiplicity=None, plot=False,
                        num_confs=None, confs_to_return=None, well_tolerance=None, de_threshold=None,
                        max_confs_num=None, smeared_scan_res=None, combination_threshold=None, force_field='MMFF94',
                        max_combination_iterations=None, determine_h_bonds=False):
    """
    Generate conformers for (non-TS) species.

    Starting from a list of RMG Molecules.
    (resonance structures are assumed to have already been generated and included in the molecule list)
    Find a conformer using MMFF94, then parameterize an Amber FF specific for the species

    Args:
        mol_list (list or Molecule): Molecule objects to consider (or Molecule, resonance structures will be generated)
        label (str): The species' label
        torsions (list, optional): A list of all possible torsions in the molecule. Will be determined if not given.
        tops (list, optional): A list of tops corresponding to torsions. Will be determined if not given.
        xyzs (list), optional: A list of user guess xyzs, each in string format, that will also be taken into account.
        charge (int, optional): The species charge. Used to perceive a molecule from xyz.
        multiplicity (int, optional): The species multiplicity. Used to perceive a molecule from xyz.
        plot (bool, optional): Whether to plot.
        num_confs (int, optional): The number of conformers to generate. Determined automatically if not given.
        well_tolerance (float, optional): The required precision (in degrees) around which to center a well's mean.
        confs_to_return (int, optional): The number of conformers to return (the rest will be written to a file).
        de_threshold (float, optional): Energy threshold (in kJ/mol) above which wells will not be considered.
        max_confs_num (int, optional): The maximal number of conformers to consider when generating well combinations.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        combination_threshold (int, optional): A threshold below which all combinations will be generated.
        force_field (str, unicode, optional): The type of force field to use (MMFF94, MMFF94s, UFF, GAFF, fit).
                                                'fit' will first run MMFF94, than fit a custom Amber FF to the species.
        determine_h_bonds (bool, optional): Whether to determine add. conformers w/ hydrogen bonds from the lowest one.

    Returns:
        list: Lowest conformers (number of entries according to confs_to_return)

    Raises:
        ConformerError: If something goes wrong.

    Todo:
        * return the time, number of high E wells per torsion, number of symmetric torsions identified,
        * if tor_angles (after removing symmetric rotors) is empty, run a "normal" conformational test (Phthalocyanine, naphthalene)
        * Determine cases for which this approach won't work: when there are too many smeared torsions (Ciraparantag)
        * hydrogen bonds to the lowest conformer
        * chair boat configurations of cyclohexane (https://en.wikipedia.org/wiki/Cyclohexane_conformation)
        * Look at the other conformer methods from the roadmap
        * For the lowest one implement H bonds (use flag), DFT them as well (add an H-bonds flag)
        * change chiral centers and re-run
    """
    t0 = time.time()

    confs_to_return = confs_to_return or CONFS_TO_RETURN
    max_confs_num = max_confs_num or MAX_CONFS_NUM
    max_combination_iterations = max_combination_iterations or MAX_COMBINATION_ITERATIONS
    combination_threshold = combination_threshold or COMBINATION_THRESHOLD

    if isinstance(mol_list, Molecule):
        mol_list = [mol for mol in mol_list.generate_resonance_structures() if mol.reactive]
    if not isinstance(mol_list, list):
        raise ConformerError('The `mol_list` argument must be a list, got {0}'.format(type(mol_list)))
    for mol in mol_list:
        if not isinstance(mol, Molecule):
            raise ConformerError('Each entry in the `mol_list` argument must be an RMG Molecule object, '
                                 'got {0}'.format(type(mol)))
    mol_list = [update_mol(mol) for mol in mol_list]

    if torsions is None or tops is None:
        torsions, tops = determine_rotors(mol_list)
    conformers = generate_force_field_conformers(mol_list=mol_list, label=label, xyzs=xyzs, torsion_num=len(torsions),
                                                 charge=charge, multiplicity=multiplicity, num_confs=num_confs,
                                                 force_field=force_field)
    conformers = determine_dihedrals(conformers, torsions)
    conformers, hypothetical_num_comb = deduce_new_conformers(conformers, torsions, tops, mol_list,
                                                              de_threshold, smeared_scan_res, force_field=force_field,
                                                              max_combination_iterations=max_combination_iterations,
                                                              combination_threshold=combination_threshold, plot=plot)
    confs_to_return = min(confs_to_return, hypothetical_num_comb)
    lowest_confs = get_lowest_confs(conformers, n=confs_to_return)

    execution_time = time.time() - t0
    t, s = divmod(execution_time, 60)
    t, m = divmod(t, 60)
    d, h = divmod(t, 24)
    days = '{0} days and '.format(int(d)) if d else ''
    time_str = '{days}{hrs:02d}:{min:02d}:{sec:02d}'.format(days=days, hrs=int(h), min=int(m), sec=int(s))
    if execution_time > 30:
        logger.info('Conformer execution_time using {0}: {1}'.format(force_field, time_str))

    return lowest_confs, conformers  # todo: process: dump conformers to {spc}_conformer_search_space.yaml
                                     # todo: and lowest_conformers to {spc}_considered_conformers.yaml
                                     # todo: modify conformer readers as well


def deduce_new_conformers(conformers, torsions, tops, mol_list, de_threshold=None, smeared_scan_res=None,
                          force_field='MMFF94', max_combination_iterations=25, combination_threshold=1000, plot=False):
    """
    By knowing the existing torsion wells, get the geometries of all important conformers.

    Validate that atoms don't collide in the generated conformers (don't consider ones where they do)

    Args:
        conformers (list): Entries are conformer dictionaries.
        torsions (list): A list of all possible torsion angles in the molecule, each torsion angles list is sorted.
        tops (list): A list of tops corresponding to torsions.
        mol_list (list): A list of RMG Molecule objects.
        de_threshold (float): An energy threshold (in kJ/mol) above which wells in a torsion will not be considered.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        force_field (str, unicode, optional): The type of force field to use.
        max_combination_iterations (int, optional): The max num of times to iteratively search for the lowest conformer.
        combination_threshold (int, optional): A threshold below which all combinations will be generated.
        plot (bool, optional): Whether or not to show plots.

    Returns:
        list: New conformer combinations, entries are conformer dictionaries.
        int: Number of all possible combinations
    """
    de_threshold = de_threshold or DE_THRESHOLD
    smeared_scan_res = smeared_scan_res or SMEARED_SCAN_RESOLUTIONS
    torsion_angles = get_torsion_angles(conformers, torsions)  # get all wells per torsion
    mol = mol_list[0]

    symmetries = dict()
    for torsion, top in zip(torsions, tops):
        # identify symmetric torsions so we don't bother considering them in the conformational combinations
        symmetry = determine_torsion_symmetry(torsion, top, mol_list, torsion_angles[tuple(torsion)])
        symmetries[tuple(torsion)] = symmetry
    # logger.info('Identified {0} symmetric wells'.format(len([s for s in symmetries.values() if s > 1])))

    torsions_sampling_points, wells_dict = dict(), dict()
    for tor, tor_angles in torsion_angles.items():
        torsions_sampling_points[tor], wells_dict[tor] =\
            determine_torsion_sampling_points(tor_angles, smeared_scan_res=smeared_scan_res,
                                              symmetry=symmetries[tor])

    if plot:
        arc.plotter.plot_torsion_angles(torsion_angles, torsions_sampling_points, wells_dict=wells_dict)

    hypothetical_num_comb = 1
    for points in torsions_sampling_points.values():
        hypothetical_num_comb *= len(points)
    logger.info('\nHypothetical number of combinations: {0:.2E}'.format(hypothetical_num_comb))

    # get the lowest conformer as the base xyz for further processing
    lowest_conf = get_lowest_confs(conformers, n=1)[0]

    # split torsions_sampling_points into two lists, use combinations only for those with multiple sampling points
    single_tors, multiple_tors, single_sampling_point, multiple_sampling_points = list(), list(), list(), list()
    multiple_sampling_points_dict = dict()  # used for plotting an energy "scan"
    for tor, points in torsions_sampling_points.items():
        if len(points) == 1:
            single_tors.append(tor)
            single_sampling_point.append((points[0]))
        else:
            multiple_sampling_points_dict[tor] = points
            multiple_tors.append(tor)
            multiple_sampling_points.append(points)

    new_conformers = list()  # will be returned
    # set symmetric (single well) torsions to the mean of the well
    base_xyz = lowest_conf['xyz']  # base_xyz is modified within the loop in each iteration
    logger.info('original lowest conformer:')
    arc.plotter.show_sticks(base_xyz)
    for torsion, dihedral in zip(single_tors, single_sampling_point):
        conf, rd_mol, index_map = converter.rdkit_conf_from_mol(mol, base_xyz)
        rd_tor_map = [index_map[i - 1] for i in torsion]  # convert the atom indices in the torsion to RDKit indices
        base_xyz = converter.set_rdkit_dihedrals(conf, rd_mol, index_map, rd_tor_map, deg_abs=dihedral)

    if hypothetical_num_comb > combination_threshold:
        logger.info('hypothetical_num_comb > {0}'.format(combination_threshold))
        lowest_conf_i = None
        # don't generate all combinations, there are simply too many
        # iteratively filter by energy and atom collisions
        for i in range(max_combination_iterations):
            logger.debug('iteration {0}'.format(i))
            newest_conformers_dict, newest_conformer_list = dict(), list()  # conformers from the current iteration
            for tor, sampling_points in zip(multiple_tors, multiple_sampling_points):
                energies, xyzs = change_dihedrals_and_force_field_it(mol, xyz=base_xyz, torsions=[tor],
                                                                     new_dihedrals=[[sp] for sp in sampling_points],
                                                                     force_field=force_field)
                newest_conformers_dict[tor] = list()  # keys are torsions for plotting
                for energy, xyz, dihedral in zip(energies, xyzs, sampling_points):
                    exists = False
                    for conf in new_conformers + newest_conformer_list:
                        if compare_xyz(xyz, conf['xyz']):
                            exists = True
                            break
                    if xyz is not None:
                        conformer = {'index': len(conformers) + len(new_conformers) + len(newest_conformer_list),
                                     'xyz': xyz,
                                     'FF energy': round(energy, 3),
                                     'source': 'Changing dihedrals on most stable conformer, iteration {0}'.format(i),
                                     'torsion': tor,
                                     'dihedral': round(dihedral, 2)}
                        newest_conformers_dict[tor].append(conformer)
                        if not exists:
                            newest_conformer_list.append(conformer)
                    else:
                        # if xyz is None, atoms have collided
                        logger.debug('\n\natoms colliding for torsion {0} and dihedral {1}:'.format(tor, dihedral))
                        logger.debug(xyz)
                        logger.debug('\n\n')
            new_conformers.extend(newest_conformer_list)
            if not newest_conformer_list:
                newest_conformer_list = [lowest_conf_i]
            if force_field != 'gromacs':
                lowest_conf_i = get_lowest_confs(newest_conformer_list, n=1)[0]
                if plot:
                    logger.info('comparing lowest xyz to base xyz:')
                    arc.plotter.show_sticks(lowest_conf_i['xyz'])
                    logger.info(compare_xyz(lowest_conf['xyz'], lowest_conf_i['xyz']))
                if lowest_conf_i['FF energy'] == lowest_conf['FF energy']\
                        and compare_xyz(lowest_conf['xyz'], lowest_conf_i['xyz']):
                    break
                elif lowest_conf_i['FF energy'] < lowest_conf['FF energy']:
                    lowest_conf = lowest_conf_i

        if plot:
            num_comb = arc.plotter.plot_torsion_angles(torsion_angles, multiple_sampling_points_dict,
                                                       wells_dict=wells_dict, e_conformers=newest_conformers_dict,
                                                       de_threshold=de_threshold)
            if num_comb is not None:
                logger.info('Number of combinations after reduction: {0:.2E}'.format(num_comb))

    else:
        logger.info('hypothetical_num_comb < {0}'.format(combination_threshold))
        # just generate all combinations and get their FF energies

        # generate sampling points combinations
        product_combinations = list(product(*multiple_sampling_points))

        if multiple_tors:
            energies, xyzs = change_dihedrals_and_force_field_it(mol, xyz=base_xyz, torsions=multiple_tors,
                                                                 new_dihedrals=product_combinations, optimize=False,
                                                                 force_field=force_field)
            for energy, xyz in zip(energies, xyzs):
                if xyz is not None:
                    new_conformers.append({'index': len(conformers) + len(new_conformers),
                                           'xyz': xyz,
                                           'FF energy': energy,
                                           'source': 'Generated all combinations from scan map'})
        else:
            # no multiple torsions (all torsions are symmetric or no torsions in the molecule), this is a trivial case
            new_conformers.append({'index': len(conformers) + len(new_conformers),
                                   'xyz': converter.get_xyz_string(coord=base_xyz, mol=mol),
                                   'FF energy': lowest_conf['FF energy'],
                                   'source': 'Generated all combinations from scan map'})
    return new_conformers, hypothetical_num_comb


def generate_force_field_conformers(mol_list, label, torsion_num, charge, multiplicity, xyzs=None, num_confs=None,
                                    force_field='MMFF94'):
    """
    Generate conformers using RDKit and Open Babel and optimize them using a force field
    Also consider user guesses in `xyzs`

    Args:
        mol_list (list): Entries are Molecule objects representing resonance structures of a chemical species.
        label (str, unicode): The species label.
        xyzs (list, optional): Entries are xyz coordinates in string format, given as initial guesses.
        torsion_num (int): The number of torsions identified in the molecule.
        charge (int): The net charge of the species.
        multiplicity (int): The species spin multiplicity.
        num_confs (int, optional): The number of conformers to generate.
        force_field (str, unicode, optional): The type of force field to use.

    Returns:
        list: Entries are conformer dictionaries.

    Raises:
        ConformerError: If xyzs is given and it is not a list, or its entries are not strings.
    """
    conformers = list()
    number_of_heavy_atoms = len([atom for atom in mol_list[0].atoms if atom.isNonHydrogen()])
    num_confs = num_confs or determine_number_of_conformers_to_generate(heavy_atoms=number_of_heavy_atoms,
                                                                        torsion_num=torsion_num, label=label)
    logger.info('Species {0} has {1} heavy atoms and {2} torsions. Using {3} random conformers.'.format(
        label, number_of_heavy_atoms, torsion_num, num_confs))
    for mol in mol_list:
        ff_xyzs, ff_energies = list(), list()
        try:
            ff_energies, ff_xyzs = get_force_field_energies(mol, num_confs=num_confs, return_xyz_strings=True,
                                                            force_field=force_field)
        except ValueError as e:
            logger.warning('Could not generate conformers for {0}, failed with: {1}'.format(label, e.message))
        if ff_xyzs:
            for xyz, energy in zip(ff_xyzs, ff_energies):
                conformers.append({'xyz': xyz,
                                   'index': len(conformers),
                                   'FF energy': energy,
                                   'source': force_field})
    # User guesses
    if xyzs is not None and xyzs:
        if not isinstance(xyzs, list):
            raise ConformerError('The xyzs argument must be a list, got {0}'.format(type(xyzs)))
        for xyz in xyzs:
            if not isinstance(xyz, (str, unicode)):
                raise ConformerError('Each entry in xyzs must be a string, got {0}'.format(type(xyz)))
            s_mol, b_mol = converter.molecules_from_xyz(xyz, multiplicity=multiplicity, charge=charge)
            conformers.append({'xyz': xyz,
                               'index': len(conformers),
                               'FF energy': get_force_field_energies(mol=b_mol or s_mol, xyz=xyz,
                                                                     return_xyz_strings=True, optimize=True,
                                                                     force_field=force_field),
                               'source': 'User Guess'})
    return conformers


def change_dihedrals_and_force_field_it(mol, xyz, torsions, new_dihedrals, optimize=True, return_xyz_strings=True,
                                        force_field='MMFF94'):
    """
    Change dihedrals of specified torsions according to the new dihedrals specified, and get FF energies.

    Example:
        torsions = [(1, 2, 3, 4), (9, 4, 7, 1)]
        new_dihedrals = [[90, 120], [90, 300], [180, 270], [30, 270]]
        This will calculate the energy of the original conformer (defined using `xyz`).
        We iterate through new_dihedrals. The torsions are set accordingly and the energy and xyz of the newly
        generated conformer are kept.
        We assume that each list entry in new_dihedrals is of the length of the torsions list (2 in the example).

    Args:
        mol (Molecule): The RMG molecule with the connectivity information.
        xyz (str, unicode, or list): The base 3D geometry to be changed, in either string or array format.
        torsions (list): Entries are torsion tuples for which the dihedral will be changed relative to xyz.
        new_dihedrals (list): Entries are same size lists of dihedral angles (floats) corresponding to the torsions.
        optimize (bool, optional): Whether to optimize the generated conformer using FF. True to optimize.
        return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
        force_field (str, unicode, optional): The type of force field to use.

    Returns:
        list: The conformer FF energies corresponding to the list of dihedrals.
        list: The conformer xyz geometries corresponding to the list of dihedrals.
    """
    if isinstance(xyz, (str, unicode)):
        xyz = converter.get_xyz_matrix(xyz)[0]

    if torsions is None or new_dihedrals is None:
        energy, xyz = get_force_field_energies(mol=mol, xyz=xyz, optimize=True, return_xyz_strings=return_xyz_strings,
                                               force_field=force_field)
        return energy, xyz

    energies, xyzs = list(), list()
    # make sure new_dihedrals is a list of lists (or tuples):
    if isinstance(new_dihedrals, (int, float)):
        new_dihedrals = [[new_dihedrals]]
    if isinstance(new_dihedrals, list) and not isinstance(new_dihedrals[0], (list, tuple)):
        new_dihedrals = [new_dihedrals]

    for dihedrals in new_dihedrals:
        for torsion, dihedral in zip(torsions, dihedrals):
            conf, rd_mol, indx_map = converter.rdkit_conf_from_mol(mol, xyz)
            rd_tor_map = [indx_map[i - 1] for i in torsion]  # convert the atom indices in the torsion to RDKit indices
            xyz = converter.set_rdkit_dihedrals(conf, rd_mol, indx_map, rd_tor_map, deg_abs=dihedral)
            if force_field != 'gromacs':
                energy, xyz_ = get_force_field_energies(mol=mol, xyz=xyz, optimize=optimize, force_field=force_field,
                                                        return_xyz_strings=return_xyz_strings)
                if energy and xyz_:
                    print(energy)
                    print(xyz_)
                    energies.append(energy[0])
                    xyzs.append(xyz_[0])
            else:
                energies.append(None)
                xyzs.append(xyz)
    return energies, xyzs


def determine_rotors(mol_list):
    """
    Determine possible unique rotors in the species to be treated as hindered rotors

    Args:
        mol_list (list): Localized structures (Molecule objects) by which all rotors will be determined

    Returns:
        list: A list of indices of scan pivots
        list: A list of indices of top atoms (including one of the pivotal atoms) corresponding to the torsions
    """
    torsions, tops = list(), list()
    for mol in mol_list:
        rotors = find_internal_rotors(mol)
        for new_rotor in rotors:
            for existing_torsion in torsions:
                if existing_torsion == new_rotor['scan']:
                    break
            else:
                torsions.append(new_rotor['scan'])
                tops.append(new_rotor['top'])
    return torsions, tops


def determine_number_of_conformers_to_generate(heavy_atoms, torsion_num, label, minimalist=False):
    """
    Determine the number of conformers to generate using molecular mechanics

    Args:
        heavy_atoms (int): The number of heavy atoms in the molecule.
        torsion_num (int): The number of potential torsions in the molecule.
        label (str, unicode): The species' label.
        minimalist (bool, optional): Whether to return a small number of conformers, useful when this is just a guess
                                     before fitting a force field. True to be minimalistic.

    Returns:
        int: number of conformers to generate.

    Raises:
        ConformerError: If the number of conformers to generate cannot be determined.
    """
    if isinstance(torsion_num, list):
        torsion_num = len(torsion_num)

    for heavy_range, num_confs_1 in CONFS_VS_HEAVY_ATOMS.items():
        if heavy_range[1] == 'inf' and heavy_atoms >= heavy_range[0]:
            break
        elif heavy_range[1] >= heavy_atoms >= heavy_range[0]:
            break
    else:
        raise ConformerError('Could not determine the number of conformers to generate according to the number '
                             'of heavy atoms ({heavy}) in {label}. The CONFS_VS_HEAVY_ATOMS dictionary might be '
                             'corrupt, got:\n {d}'.format(heavy=heavy_atoms, label=label, d=CONFS_VS_HEAVY_ATOMS))

    for torsion_range, num_confs_2 in CONFS_VS_TORSIONS.items():
        if torsion_range[1] == 'inf' and torsion_num >= torsion_range[0]:
            break
        elif torsion_range[1] >= torsion_num >= torsion_range[0]:
            break
    else:
        raise ConformerError('Could not determine the number of conformers to generate according to the number '
                             'of torsions ({torsion_num}) in {label}. The CONFS_VS_TORSIONS dictionary might be '
                             'corrupt, got:\n {d}'.format(torsion_num=torsion_num, label=label, d=CONFS_VS_TORSIONS))

    if minimalist:
        num_confs = min(num_confs_1, num_confs_2, 250)
    else:
        num_confs = max(num_confs_1, num_confs_2)

    return num_confs


def determine_dihedrals(conformers, torsions):
    """
    For each conformer in `conformers` determine the respective dihedrals.

    Args:
        conformers (list): Entries are conformer dictionaries.
        torsions (list): All possible torsions in the molecule.

    Returns:
        list: Entries are conformer dictionaries.
    """
    for conformer in conformers:
        if isinstance(conformer['xyz'], (str, unicode)):
            coord = converter.get_xyz_matrix(conformer['xyz'])[0]
        else:
            coord = conformer['xyz']
        if 'torsion_dihedrals' not in conformer or not conformer['torsion_dihedrals']:
            conformer['torsion_dihedrals'] = dict()
            for torsion in torsions:
                angle = calculate_dihedral_angle(coord=coord, torsion=torsion)
                conformer['torsion_dihedrals'][tuple(torsion)] = angle
    return conformers


def determine_torsion_sampling_points(torsion_angles, smeared_scan_res=None, symmetry=1):
    """
    Determine how many points to consider in each well of a torsion for conformer combinations

    Args:
        torsion_angles (list): Well angles in the torsion.
        smeared_scan_res (float, optional): The resolution (in degrees) for scanning smeared wells.
        symmetry (int, optional): The torsion symmetry number.

    Returns:
        list: Sampling points for the torsion.
        list: Each entry is a well dictionary with the keys:
             'start_idx', 'end_idx', 'start_angle', 'end_angle', 'angles'.
    """
    smeared_scan_res = smeared_scan_res or SMEARED_SCAN_RESOLUTIONS
    sampling_points = list()
    wells = get_wells(torsion_angles, blank=20)
    for i, well in enumerate(wells):
        width = abs(well['end_angle'] - well['start_angle'])
        mean = sum(well['angles']) / len(well['angles'])
        if width <= 2 * smeared_scan_res:
            sampling_points.append(mean)
        else:
            num = int(width / smeared_scan_res)
            padding = abs(mean - well['start_angle'] - ((num - 1) * smeared_scan_res) / 2)
            sampling_points.extend([padding + well['angles'][0] + smeared_scan_res * j for j in range(int(num))])
        if symmetry > 1 and i == len(wells) / symmetry - 1:
            break
    return sampling_points, wells


def determine_torsion_symmetry(torsion, top1, mol_list, torsion_scan):
    """
    Check whether a torsion is symmetric

    If a torsion well is "well defined" and not smeared, it could be symmetric
    check the groups attached to the rotor pivots to determine whether it is indeed symmetric
    We don't care about the actual rotor symmetry number here, since we plan to just use the first well
    (they're all the same)

    Args:
        torsion (list): A list of four atom indices that define the torsion.
        top1 (list): A list of atom indices on one side of the torsion, including the pivotal atom.
        mol_list (list): A list of molecules.
        torsion_scan (list): The angles corresponding to this torsion from all conformers.

    Returns:
        int: The rotor symmetry number.

    Todo:
        * multiply symmetries instead or returning them. think of Cc1ccccc1 or C[N+](=O)[O-] with symm = 3 * 2
        but break the for tops loop to continue to the next if found to one of the tops
        * Bug: torsion (14, 7, 17, 22) in CJ gets a symmetry number of 2 (should be 1)
        * `torsion` isn't needed here
    """
    # pnt = False
    # if torsion == [14, 7, 17, 22]:
    #     pnt = True
    #     logger.info(torsion)
    #     logger.info(top1)
    mol = mol_list[0]
    top2 = [i + 1 for i in range(len(mol.atoms)) if i + 1 not in top1]
    for top in [top1, top2]:
        # A quick bypass for methyl rotors which are too common:
        if len(top) == 4 and mol.atoms[top[0] - 1].isCarbon() \
                and all([mol.atoms[top[i] - 1].isHydrogen() for i in range(1, 4)]):
            return 3
        # A quick bypass for benzene rings:
        if len(top) == 11 and sum([mol.atoms[top[i] - 1].isCarbon() for i in range(11)]) == 6 \
                and sum([mol.atoms[top[i] - 1].isHydrogen() for i in range(11)]) == 5:
            return 2
    # treat the torsion list as cyclic, search for at least two blank parts of at least 60 degrees each
    # if the means of all data parts of the scan are uniformly scattered, the torsion might be symmetric
    wells = get_wells(angles=torsion_scan, blank=60)

    distances, well_widths = list(), list()
    for i in range(len(wells)):
        well_widths.append(abs(wells[i]['end_angle'] - wells[i]['start_angle']))
        if i > 0:
            distances.append(int(round(abs(wells[i]['start_angle'] - wells[i - 1]['end_angle'])) / 10) * 10)
    mean_well_width = sum(well_widths) / len(well_widths)
    if len(wells) in [2, 3] and all([distance == distances[0] for distance in distances]) \
            and all([abs(width - mean_well_width) / mean_well_width < determine_well_width_tolerance(mean_well_width)
                     for width in well_widths]):
        # All well distances and widths are equal. The torsion scan might be symmetric, check the groups
        for top in [top1, top2]:
            groups = list()
            grp_idx = list()
            groups_indices = list()
            for atom in mol.atoms[top[0] - 1].edges.keys():
                if mol.vertices.index(atom) + 1 in top:
                    # loop atoms adjacent to the pivot if they are in top
                    atom_indices = list()
                    explored_atom_list, atom_list_to_explore1, atom_list_to_explore2 = \
                        [mol.atoms[top[0] - 1]], [atom], []
                    while len(atom_list_to_explore1 + atom_list_to_explore2):
                        for atom1 in atom_list_to_explore1:
                            atom_indices.append(mol.vertices.index(atom1))
                            for atom2 in atom.edges.keys():
                                if atom2.isHydrogen():
                                    # append H w/o further exploring
                                    atom_indices.append(mol.vertices.index(atom2))
                                elif atom2 not in explored_atom_list and atom2 not in atom_list_to_explore2:
                                    atom_list_to_explore2.append(atom2)  # explore it further
                            explored_atom_list.append(atom1)  # mark as explored
                        atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
                    groups.append(to_group(mol, atom_indices))
                    grp_idx.append(atom_indices)
                    groups_indices.append([g + 1 for g in atom_indices])
            # hard-coding for NO2/NS2 groups, since the two O or S atoms have different atom types in each localized
            # structure, hence are not isomorphic
            if len(top) == 3 and mol.atoms[top[0] - 1].atomType.label == 'N5dc' \
                    and (all([mol.atoms[top[i] - 1].atomType.label in ['O2d', 'O0sc'] for i in [1, 2]])
                         or all([mol.atoms[top[i] - 1].atomType.label in ['S2d', 'S0sc'] for i in [1, 2]])):
                return 2
            # all other groups:
            if not mol.atoms[top[0] - 1].lonePairs > 0 and not mol.atoms[top[0] - 1].radicalElectrons > 0 \
                    and all([groups[0].isIsomorphic(group) for group in groups[1:]]):
                # if pnt:
                #     logger.info('**** why symmetric??')
                #     logger.info('len groups ', len(groups))
                #     for g in grp_idx:
                #         logger.info(g)
                return len(groups)
    return 1


def determine_well_width_tolerance(mean_width):
    """
    Determine the tolerance by which well wedths are determined to be nearly equal

    Fitted to a polynomial trend line for the following data of (mean, tolerance) pairs:
    (100, 0.11), (60, 0.13), (50, 0.15), (25, 0.25), (5, 0.50), (1, 0.59)

    Args:
        mean_width (float): The mean well width in degrees.

    Returns:
        float: The tolerance.
    """
    if mean_width > 100:
        return 0.1
    tol = -1.695e-10 * mean_width ** 5 + 6.209e-8 * mean_width ** 4 - 8.855e-6 * mean_width ** 3 \
        + 6.446e-4 * mean_width ** 2 - 2.610e-2 * mean_width + 0.6155
    return tol


def get_lowest_confs(confs, n=1, energy='FF energy'):
    """
    Get the most stable conformer

    Args:
        confs (list): Entries are either conformer dictionaries or a length two list of xyz coordinates and energy.
        n (int): Number of lowest conformers to return.
        energy (str, unicode, optional): The energy attribute to search by. Currently only 'FF energy' is supported.

    Returns:
        list: Conformer dictionaries.
    """
    if not confs:
        raise ConformerError('get_lowest_confs() got no conformers')
    if isinstance(confs[0], dict):
        conformer_list = [conformer for conformer in confs if energy in conformer and conformer[energy]]
        return nsmallest(n, conformer_list, key=lambda conf: conf[energy])
    elif isinstance(confs[0], list):
        return nsmallest(n, confs, key=lambda conf: conf[1])
    else:
        raise ConformerError('confs could either be a list of dictionaries or a list of lists. '
                             'Got a list of {0}'.format(type(confs[0])))


def get_torsion_angles(conformers, torsions):
    """
    Populate each torsion pivots with all available angles from the generated conformers

    Args:
        conformers (list): The conformers from which to extract the angles.
        torsions (list): The torsions to consider.

    Returns:
        dict: The torsion angles. Keys are torsion tuples, values are lists of all corresponding angles from conformers.
    """
    torsion_angles = dict()
    if not any(['torsion_dihedrals' in conformer for conformer in conformers]):
        raise ConformerError('Could not determine dihedral torsion angles. '
                             'Consider calling `determine_dihedrals()` first.')
    for conformer in conformers:
        if 'torsion_dihedrals' in conformer and conformer['torsion_dihedrals']:
            for torsion in torsions:
                if tuple(torsion) not in torsion_angles:
                    torsion_angles[tuple(torsion)] = list()
                torsion_angles[tuple(torsion)].append(conformer['torsion_dihedrals'][tuple(torsion)])
    for tor in torsion_angles.keys():
        torsion_angles[tor].sort()
    return torsion_angles


def get_force_field_energies(mol, num_confs=None, xyz=None, force_field='MMFF94', return_xyz_strings=True,
                             optimize=True):
    """
    Determine force field energies using RDKit.
    If num_confs is given, random 3D geometries will be generated. If xyz is given, it will be directly used instead.
    The coordinates are returned in the order of atoms in mol.

    Args:
        mol (Molecule): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (string or list, optional): A 3D coordinates guess in either a string or an array format.
        force_field (str, unicode, optional): The type of force field to use.
        return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
        optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.
        mix (bool, optional): Whether to mix RDKit and open babel, to get the best of both worlds. True to mix.

    Returns:
        list: Entries are xyz coordinates.
        list: Entries are the FF energies (in kJ/mol).
    """
    if force_field.lower() in ['mmff94', 'mmff94s', 'uff']:
        rd_mol, rd_index_map = embed_rdkit(mol, num_confs=num_confs, xyz=xyz)
        xyzs, energies = rdkit_force_field(rd_mol, rd_index_map=rd_index_map, mol=mol, force_field=force_field,
                                           return_xyz_strings=return_xyz_strings, optimize=optimize)
    elif force_field.lower() in ['gaff', 'mmff94', 'mmff94s', 'uff', 'ghemical']:
        xyzs, energies = mix_rdkit_and_openbabel_force_field(mol, num_confs=num_confs, xyz=xyz, force_field=force_field,
                                                             return_xyz_strings=True)
    else:
        raise ConformerError('Unrecognized force field. Should be either MMFF94, MMFF94s, UFF, Ghemical, '
                             'or GAFF. Got: {0}'.format(force_field))
    return energies, xyzs


def mix_rdkit_and_openbabel_force_field(mol, num_confs=None, xyz=None, force_field='GAFF', return_xyz_strings=True):
    """
    Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical)
    Use RDKit to generate the random conformers (open babel isn't good enough),
    but use open babel to optimize them (RDKit doesn't have GAFF)

    Args:
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (string or list, optional): The 3D coordinates in either a string or an array format.
        force_field (str, unicode, optional): The type of force field to use.
        return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
        method (str, unicode, optional): The conformer searching method to use in open babel.

    Returns:
        list: Entries are optimized xyz's in a list format.
        list: Entries float numbers representing the energies in kJ/mol.
    """
    xyzs, energies = list(), list()
    rd_mol, rd_index_map = embed_rdkit(mol, num_confs=num_confs, xyz=xyz)
    unoptimized_xyzs = list()
    for i in range(rd_mol.GetNumConformers()):
        conf, xyz = rd_mol.GetConformer(i), list()
        for j in range(conf.GetNumAtoms()):
            pt = conf.GetAtomPosition(j)
            xyz.append([pt.x, pt.y, pt.z])
        xyz = [xyz[rd_index_map[j]] for j, _ in enumerate(xyz)]  # reorder
        unoptimized_xyzs.append(xyz)  # in array form

    for xyz in unoptimized_xyzs:
        xyzs_, energies_ = openbabel_force_field(mol, num_confs, xyz=xyz, force_field=force_field,
                                                 return_xyz_strings=return_xyz_strings)
        xyzs.extend(xyzs_)
        energies.extend(energies_)
    return xyzs, energies


def openbabel_force_field(mol, num_confs=None, xyz=None, force_field='GAFF', return_xyz_strings=True,
                          method='diverse'):
    """
    Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical)

    Args:
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (list, optional): The 3D coordinates in an array format.
        force_field (str, unicode, optional): The type of force field to use.
        return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
        method (str, unicode, optional): The conformer searching method to use in open babel.
                                         For method description, see http://openbabel.org/dev-api/group__conformer.shtml

    Returns:
        list: Entries are optimized xyz's in a list format.
        list: Entries float numbers representing the energies in kJ/mol.
    """
    xyzs, energies = list(), list()
    ff = ob.OBForceField.FindForceField(force_field)

    if xyz is not None:
        if isinstance(xyz, (str, unicode)):
            xyz = converter.get_xyz_matrix(xyz)[0]
        # generate an open babel molecule
        obmol = ob.OBMol()
        atoms = mol.vertices
        ob_atom_ids = dict()  # dictionary of OB atom IDs
        for i, atom in enumerate(atoms):
            a = obmol.NewAtom()
            a.SetAtomicNum(atom.number)
            a.SetVector(xyz[i][0], xyz[i][1], xyz[i][2])  # assume xyz is ordered like mol; line not in in toOBMol
            if atom.element.isotope != -1:
                a.SetIsotope(atom.element.isotope)
            a.SetFormalCharge(atom.charge)
            ob_atom_ids[atom] = a.GetId()
        orders = {1: 1, 2: 2, 3: 3, 4: 4, 1.5: 5}
        for atom1 in mol.vertices:
            for atom2, bond in atom1.edges.items():
                if bond.isHydrogenBond():
                    continue
                index1 = atoms.index(atom1)
                index2 = atoms.index(atom2)
                if index1 < index2:
                    obmol.AddBond(index1 + 1, index2 + 1, orders[bond.order])

        # optimize
        ff.Setup(obmol)
        ff.SetLogLevel(0)
        ff.SetVDWCutOff(6.0)  # The VDW cut-off distance (default=6.0)
        ff.SetElectrostaticCutOff(10.0)  # The Electrostatic cut-off distance (default=10.0)
        ff.SetUpdateFrequency(10)  # The frequency to update the non-bonded pairs (default=10)
        ff.EnableCutOff(False)  # Use cut-off (default=don't use cut-off)
        # ff.SetLineSearchType('Newton2Num')
        ff.SteepestDescentInitialize()  # ConjugateGradientsInitialize
        v = 1
        while v:
            v = ff.SteepestDescentTakeNSteps(1)  # ConjugateGradientsTakeNSteps
            if ff.DetectExplosion():
                raise ConformerError('Force field {0} exploded with method {1}'.format(force_field, 'SteepestDescent'))
        ff.GetCoordinates(obmol)

    elif num_confs is not None:
        obmol, ob_atom_ids = toOBMol(mol, returnMapping=True)
        pybmol = pyb.Molecule(obmol)
        pybmol.make3D()
        ff.Setup(obmol)

        if method.lower() == 'weighted':
            ff.WeightedRotorSearch(num_confs, 2000)
        elif method.lower() == 'random':
            ff.RandomRotorSearch(num_confs, 2000)
        elif method.lower() == 'diverse':
            rmsd_cutoff = 0.5
            energy_cutoff = 50.
            confab_verbose = False
            ff.DiverseConfGen(rmsd_cutoff, num_confs, energy_cutoff, confab_verbose)
        elif method.lower() == 'systematic':
            ff.SystematicRotorSearch(num_confs)
        else:
            raise ConformerError('Could not identify method {0}'.format(method))
    else:
        raise ConformerError('Either num_confs or xyz should be given)')

    ff.GetConformers(obmol)
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('xyz')

    for i in range(obmol.NumConformers()):
        obmol.SetConformer(i)
        ff.Setup(obmol)
        xyz = '\n'.join(obconversion.WriteString(obmol).splitlines()[2:])
        if not return_xyz_strings:
            xyz = converter.get_xyz_matrix(xyz)[0]
            xyz = [xyz[ob_atom_ids[mol.atoms[j]]] for j, _ in enumerate(xyz)]  # reorder
        xyzs.append(xyz)
        energies.append(ff.Energy())
    return xyzs, energies


# def openbabel_force_field_old(mol, xyzs, force_field='gaff', return_xyz_strings=True, optimize=True):
#     """
#     Optimize conformers using a force field (GAFF, MMFF94s, MMFF94, UFF, Ghemical)
#     This function does not embed (make 3D) the molecule.
#
#     Args:
#         mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
#         xyzs (list): Each entry is a coordinats descriptor of a conformer.
#         force_field (str, unicode, optional): The type of force field to use.
#         return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
#                                              Requires mol to not be None to return string format.
#         optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.
#
#     Returns:
#         list: Entries are optimized xyz's in a list format.
#         list: Entries float numbers representing the energies.
#
#
#     `num_confs` is the number of conformers to generate
#     Uses OpenBabel to automatically generate a set of initial geometries, optimizes these geometries using MMFF94s,
#     calculates the energies using MMFF94s
#     Returns the coordinates and energies
#     """
#     # num_confs = len(xyzs)
#     ff = ob.OBForceField.FindForceField(force_field)
#     obmol, ob_atom_ids = toOBMol(mol, returnMapping=True)
#     obmol.AddConformer(1)
#     for i, xyz in enumerate(xyzs):
#         # obmol.SetConformer(i)
#         ob_xyz = [[] for j, _ in enumerate(xyz)]  # make a list of empty lists
#         for j, _ in enumerate(xyz):
#             ob_xyz[ob_atom_ids[mol.atoms[j]]] = xyz[j]  # reorder
#         obmol.SerCoordinates(ob_xyz)
#         ff.Setup(obmol)
#
#
#
#     pybmol = pyb.Molecule(obmol)
#     pybmol.make3D()
#
#     ff = ob.OBForceField.FindForceField('gaff')
#     ff.Setup(obmol)
#
#     rmsd_cutoff = 0.5
#     energy_cutoff = 50.
#     confab_verbose = False
#     ff.DiverseConfGen(rmsd_cutoff, num_confs, energy_cutoff, confab_verbose)
#     ff.GetConformers(obmol)
#
#     for i in range(num_confs):
#         obmol.SetConformer(i)
#         obconversion = ob.OBConversion()
#         obconversion.SetOutFormat('xyz')
#         xyz = '\n'.join(obconversion.WriteString(obmol).splitlines()[2:])
#         if not return_xyz_strings:
#             xyz = converter.get_xyz_matrix(xyz)[0]
#             xyz = [xyz[ob_atom_ids[mol.atoms[j]]] for j, _ in enumerate(xyz)]  # reorder
#         xyzs.append(xyz)
#     return xyzs


def embed_rdkit(mol, num_confs=None, xyz=None):
    """
    Generate random conformers (unoptimized) in RDKit

    Args:
        mol (RMG Molecule or RDKit RDMol): The molecule object with connectivity and bond order information.
        num_confs (int, optional): The number of random 3D conformations to generate.
        xyz (string or list, optional): The 3D coordinates in either a string or an array format.

    Returns:
        RDMol: An RDKIt molecule with embedded conformers.
        dict: The atom mapping dictionary.
    """
    rd_index_map = dict()
    if num_confs is None and xyz is None:
        raise ConformerError('Either num_confs or xyz must be set when calling embed_rdkit().')
    if isinstance(mol, RDMol):
        rd_mol = mol
    elif isinstance(mol, Molecule):
        rd_mol, rd_indices = mol.toRDKitMol(removeHs=False, returnMapping=True)
        for k, atom in enumerate(mol.atoms):
            index = rd_indices[atom]
            rd_index_map[index] = k
    else:
        raise ConformerError('Argument mol can be either an RMG Molecule or an RDKit RDMol object. '
                             'Got {0}'.format(type(mol)))
    if num_confs is not None:
        Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=num_confs, randomSeed=1)
        # Chem.AllChem.EmbedMultipleConfs(rd_mol, numConfs=num_confs, randomSeed=15, enforceChirality=False)
    elif xyz is not None:
        rd_conf = Chem.Conformer(rd_mol.GetNumAtoms())
        if isinstance(xyz, (str, unicode)):
            coord = converter.get_xyz_matrix(xyz)[0]
        else:
            coord = xyz
        for i in range(rd_mol.GetNumAtoms()):
            rd_conf.SetAtomPosition(i, coord[i])
        rd_mol.AddConformer(rd_conf)
    return rd_mol, rd_index_map


def rdkit_force_field(rd_mol, rd_index_map=None, mol=None, force_field='MMFF94', return_xyz_strings=True,
                      optimize=True):
    """
    Optimize RDKit conformers using a force field (MMFF94 or MMFF94s are recommended)

    Args:
        rd_mol (RDKit RDMol): The RDKit molecule with embedded conformers to optimize.
        rd_index_map (list, optional): An atom map dictionary to reorder the xyz. Requires mol to not be None.
        mol (Molecule, optional): The RMG molecule object with connectivity and bond order information.
        force_field (str, unicode, optional): The type of force field to use.
        return_xyz_strings (bool, optional): Whether to return xyz in string or array format. True for string.
                                             Requires mol to not be None to return string format.
        optimize (bool, optional): Whether to first optimize the conformer using FF. True to optimize.

    Returns:
        list: Entries are optimized xyz's in a list format.
        list: Entries float numbers representing the energies.
    """
    xyzs, energies = list(), list()
    for i in range(rd_mol.GetNumConformers()):
        if optimize:
            v, j = 1, 0
            while v and j < 200:
                v = Chem.AllChem.MMFFOptimizeMolecule(rd_mol, mmffVariant=str(force_field), confId=i,
                                                      maxIters=500, ignoreInterfragInteractions=False)
                j += 1
        mol_properties = Chem.AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant=str(force_field))
        if mol_properties is not None:
            ff = Chem.AllChem.MMFFGetMoleculeForceField(rd_mol, mol_properties, confId=i)
            if optimize:
                energies.append(ff.CalcEnergy())
            conf, xyz = rd_mol.GetConformer(i), list()
            for j in range(conf.GetNumAtoms()):
                pt = conf.GetAtomPosition(j)
                xyz.append([pt.x, pt.y, pt.z])
            if rd_index_map is not None and mol is not None:
                xyz = [xyz[rd_index_map[j]] for j, _ in enumerate(xyz)]  # reorder
            if return_xyz_strings and mol is not None:
                xyz = converter.get_xyz_string(coord=xyz, mol=mol)
            xyzs.append(xyz)
    return xyzs, energies


def get_wells(angles, blank=20):
    """
    Determine the distinct wells from a list of angles.

    Args:
        angles (list): The angles in the torsion.
        blank (int, optional): The blank space between wells.

    Returns:
        list: Each entry is a well dictionary with the keys:
             'start_idx', 'end_idx', 'start_angle', 'end_angle', 'angles'.
    """
    if not angles:
        raise ConformerError('Cannot determine wells without angles')
    new_angles = angles
    if angles[0] < -180 + blank and angles[-1] > 180 - blank:
        # relocate the first chunk of data to the end, the well seems to include the  +180/-180 degrees point
        for i, angle in enumerate(angles):
            if i > 0 and abs(angle - angles[i - 1]) > blank:
                part2 = angles[:i]
                for j, _ in enumerate(part2):
                    part2[j] += 360
                new_angles = angles[i:] + part2
                break
    wells = list()
    new_well = True
    for i in range(len(new_angles) - 1):
        if new_well:
            wells.append({'start_idx': i,
                          'end_idx': None,
                          'start_angle': new_angles[i],
                          'end_angle': None,
                          'angles': list()})
            new_well = False
        wells[-1]['angles'].append(new_angles[i])
        if abs(new_angles[i + 1] - new_angles[i]) > blank:
            # This is the last point in this well
            wells[-1]['end_idx'] = i
            wells[-1]['end_angle'] = new_angles[i]
            new_well = True
    wells[-1]['end_idx'] = len(new_angles) - 1
    wells[-1]['end_angle'] = new_angles[-1]
    wells[-1]['angles'].append(new_angles[-1])
    return wells


def check_atom_collisions(xyz):
    """
    Check whether atoms are too close to each other.

    Args:
        xyz (str, unicode): The 3D geometry.

    Returns:
         bool: True if they are colliding, False otherwise.

    Todo:
        * Atom collision distance (radii for symbols)
    """
    xyz, symbols, _, _, _ = converter.get_xyz_matrix(xyz)
    if symbols == ['H', 'H']:
        # hard-code for H2:
        if sum((xyz[0][k] - xyz[1][k]) ** 2 for k in range(3)) ** 0.5 < 0.5:
            return True
    for i, coord1 in enumerate(xyz):
        if i < len(xyz) - 1:
            for coord2 in xyz[i+1:]:
                if sum((coord1[k] - coord2[k]) ** 2 for k in range(3)) ** 0.5 < 0.9:
                    return True
    return False


def check_special_non_rotor_cases(mol, top1, top2):
    """
    Check whether one of the tops correspond to a special case which could not be rotated
    `mol` is the RMG Molecule to diagnose
    `top1` and `top2` are indices of atoms on each side of the pivots, the first index corresponds to one of the pivots
    Special cases considered are:
    - cyano groups: R-C#N
    - azide groups: N-N#N
    These cases have a 180 degree angle and torsion is meaningless, but they are identified by our methods since they
    have a single bond
    Returns `True` if this is indeed a special case which should not be treated as a rotor
    """
    for top in [top1, top2]:
        # check cyano group
        if len(top) == 2 and mol.atoms[top[0] - 1].isCarbon() and mol.atoms[top[1] - 1].isNitrogen() \
                and mol.atoms[top[1] - 1].atomType.label == 'N3t':
            return True
    for tp1, tp2 in [(top1, top2), (top2, top1)]:
        # check azide group
        if len(tp1) == 2 and mol.atoms[tp1[0] - 1].atomType.label == 'N5tc' \
                and mol.atoms[tp1[1] - 1].atomType.label == 'N3t' and mol.atoms[tp2[0] - 1].atomType.label == 'N1sc':
            return True
    return False


def find_internal_rotors(mol):
    """
    Locates the sets of indices corresponding to every internal rotor.

    Args:
        mol (Molecule): The molecule for which rotors will be determined

    Returns:
        list: All rotor dictionaries with the gaussian scan coordinates, the pivots and the smallest top.
    """
    rotors = []
    for atom1 in mol.vertices:
        if atom1.isNonHydrogen():
            for atom2, bond in atom1.edges.items():
                if atom2.isNonHydrogen() and mol.vertices.index(atom1) < mol.vertices.index(atom2) \
                        and (bond.isSingle() or bond.isHydrogenBond()) and not mol.isBondInCycle(bond):
                    if len(atom1.edges) > 1 and len(atom2.edges) > 1:  # none of the pivotal atoms are terminal
                        rotor = dict()
                        # pivots:
                        rotor['pivots'] = [mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1]
                        # top:
                        top1, top2 = [], []
                        top1_has_heavy_atoms, top2_has_heavy_atoms = False, False
                        explored_atom_list, atom_list_to_explore1, atom_list_to_explore2 = [atom2], [atom1], []
                        while len(atom_list_to_explore1 + atom_list_to_explore2):
                            for atom in atom_list_to_explore1:
                                top1.append(mol.vertices.index(atom) + 1)
                                for atom3 in atom.edges.keys():
                                    if atom3.isHydrogen():
                                        # append H w/o further exploring
                                        top1.append(mol.vertices.index(atom3) + 1)
                                    elif atom3 not in explored_atom_list and atom3 not in atom_list_to_explore2:
                                        top1_has_heavy_atoms = True
                                        atom_list_to_explore2.append(atom3)  # explore it further
                                explored_atom_list.append(atom)  # mark as explored
                            atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
                        explored_atom_list, atom_list_to_explore1, atom_list_to_explore2 = [atom1, atom2], [atom2], []
                        while len(atom_list_to_explore1 + atom_list_to_explore2):
                            for atom in atom_list_to_explore1:
                                top2.append(mol.vertices.index(atom) + 1)
                                for atom3 in atom.edges.keys():
                                    if atom3.isHydrogen():
                                        # append H w/o further exploring
                                        top2.append(mol.vertices.index(atom3) + 1)
                                    elif atom3 not in explored_atom_list and atom3 not in atom_list_to_explore2:
                                        top2_has_heavy_atoms = True
                                        atom_list_to_explore2.append(atom3)  # explore it further
                                explored_atom_list.append(atom)  # mark as explored
                            atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []
                        non_rotor = check_special_non_rotor_cases(mol, top1, top2)
                        if non_rotor:
                            continue
                        if top1_has_heavy_atoms and not top2_has_heavy_atoms:
                            rotor['top'] = top2
                        elif top2_has_heavy_atoms and not top1_has_heavy_atoms:
                            rotor['top'] = top1
                        else:
                            rotor['top'] = top1 if len(top1) <= len(top2) else top2
                        # scan:
                        rotor['scan'] = []
                        heavy_atoms = []
                        hydrogens = []
                        for atom3 in atom1.edges.keys():
                            if atom3.isHydrogen():
                                hydrogens.append(mol.vertices.index(atom3))
                            elif atom3 is not atom2:
                                heavy_atoms.append(mol.vertices.index(atom3))
                        smallest_index = len(mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        rotor['scan'].extend([mol.vertices.index(atom1) + 1, mol.vertices.index(atom2) + 1])
                        heavy_atoms = []
                        hydrogens = []
                        for atom3 in atom2.edges.keys():
                            if atom3.isHydrogen():
                                hydrogens.append(mol.vertices.index(atom3))
                            elif atom3 is not atom1:
                                heavy_atoms.append(mol.vertices.index(atom3))
                        smallest_index = len(mol.vertices)
                        if len(heavy_atoms):
                            for i in heavy_atoms:
                                if i < smallest_index:
                                    smallest_index = i
                        else:
                            for i in hydrogens:
                                if i < smallest_index:
                                    smallest_index = i
                        rotor['scan'].append(smallest_index + 1)
                        rotor['success'] = None
                        rotor['invalidation_reason'] = ''
                        rotor['times_dihedral_set'] = 0
                        rotor['scan_path'] = ''
                        rotors.append(rotor)
    return rotors


def to_group(mol, atom_indices):
    """
    This method converts a defined part of a Molecule into a Group.

    Args:
        mol (Molecule): The base molecule.
        atom_indices (list): 0-indexed atom indices corresponding to atoms in mol to be included in the group.

    Returns:
        Group: A group consisting of the desired atoms in mol.
    """
    # Create GroupAtom object for each atom in the molecule
    group_atoms = list()
    index_map = dict()  # keys are Molecule atom indices, values are Group atom indices
    for i, atom_index in enumerate(atom_indices):
        atom = mol.atoms[atom_index]
        group_atoms.append(gr.GroupAtom(atomType=[atom.atomType], radicalElectrons=[atom.radicalElectrons],
                                        charge=[atom.charge], lonePairs=[atom.lonePairs]))
        index_map[atom_index] = i
    group = gr.Group(atoms=group_atoms, multiplicity=[mol.multiplicity])
    for atom in mol.atoms:
        # Create a GroupBond for each bond between desired atoms in the molecule
        if mol.atoms.index(atom) in atom_indices:
            for bonded_atom, bond in atom.edges.items():
                if mol.atoms.index(bonded_atom) in atom_indices:
                    group.addBond(gr.GroupBond(group_atoms[index_map[mol.atoms.index(atom)]],
                                               group_atoms[index_map[mol.atoms.index(bonded_atom)]],
                                               order=[bond.order]))
    group.update()
    return group


def update_mol(mol):
    """
    Update atom types, multiplicity, and atom charges in the molecule

    Args:
        mol (Molecule): The molecule to update

    Returns:
        Molecule: the updated molecule
    """
    for atom in mol.atoms:
        atom.updateCharge()
    mol.updateAtomTypes(logSpecies=False)
    mol.updateMultiplicity()
    mol.identifyRingMembership()
    return mol


def compare_xyz(xyz1, xyz2, precision=0.1):
    """
    Compare coordinates of two conformers of the same species

    Args:
        xyz1 (list, str, unicode): Coordinates of conformer 1 in either string or array format.
        xyz2 (list, str, unicode): Coordinates of conformer 2 in either string or array format.
        precision (float, optional): The allowed difference threshold between coordinates, in Angstroms.

    Returns:
        bool: Whether the coordinates represent the same conformer, True if they do.
    """
    if isinstance(xyz1, (str, unicode)):
        xyz1 = converter.get_xyz_matrix(xyz1)[0]
    if isinstance(xyz2, (str, unicode)):
        xyz2 = converter.get_xyz_matrix(xyz2)[0]
    if not all(isinstance(xyz, list) for xyz in [xyz1, xyz2]):
        raise ConformerError('xyz1 and xyz2 can either be string or list formats, got {0} and {1}, respectively'.format(
            type(xyz1), type(xyz2)))
    for coord1, coord2 in zip(xyz1, xyz2):
        for entry1, entry2 in zip(coord1, coord2):
            if abs(entry1 - entry2) > precision:
                return False
    return True


def identify_chiral_centers(mol):
    """
    Identify the atom indices corresponding to chiral centers in a molecule

    Args:
        mol (Molecule): The molecule to be analyzed.

    Returns:
        list: Atom numbers (0-indexed) representing chiral centers in the molecule.
    """
    rd_index_map = dict()
    rd_mol, rd_indices = mol.toRDKitMol(removeHs=False, returnMapping=True)
    for k, atom in enumerate(mol.atoms):
        index = rd_indices[atom]
        rd_index_map[index] = k
    AssignStereochemistry(rd_mol, flagPossibleStereoCenters=True)
    rd_atom_chirality_flags = [atom.HasProp(str('_ChiralityPossible')) for atom in rd_mol.GetAtoms()]
    chiral_centers = list()
    for i, flag in enumerate(rd_atom_chirality_flags):
        if flag:
            chiral_centers.append(rd_index_map[i])
    return chiral_centers


def calculate_dihedral_angle(coord, torsion):
    """
    Calculate a dihedral angle. Inspired by ASE Atoms.get_dihedral().

    Args:
        coord (list): The array-format coordinates.
        torsion (list): The 4 atoms defining the dihedral angle.

    Returns:
        float: The dihedral angle.
    """
    torsion = [t - 1 for t in torsion]  # convert 1-index to 0-index
    coord = np.asarray(coord, dtype=np.float32)
    a = coord[torsion[1]] - coord[torsion[0]]
    b = coord[torsion[2]] - coord[torsion[1]]
    c = coord[torsion[3]] - coord[torsion[2]]
    bxa = np.cross(b, a)
    bxa /= np.linalg.norm(bxa)
    cxb = np.cross(c, b)
    cxb /= np.linalg.norm(cxb)
    angle = np.vdot(bxa, cxb)
    # check for numerical trouble due to finite precision:
    if angle < -1:
        angle = -1
    elif angle > 1:
        angle = 1
    angle = np.arccos(angle)
    if np.vdot(bxa, c) > 0:
        angle = 2 * np.pi - angle
    return angle * 180 / np.pi
