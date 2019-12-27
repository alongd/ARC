#!/usr/bin/env python3
# encoding: utf-8

"""
A module for manipulating vectors
"""

import math
import numpy as np

from rmgpy.molecule.molecule import Molecule

from arc.common import logger
from arc.exceptions import VectorsError
from arc.species import converter


def get_normal(v1, v2):
    """
    Calculate a normal vector using cross multiplication.

    Args:
         v1 (list): Vector 1.
         v2 (list): Vector 2.

    Returns:
        list: A normal unit vector to v1 and v2.
    """
    normal = [v1[1] * v2[2] - v2[1] * v1[2], - v1[0] * v2[2] + v2[0] * v1[2], v1[0] * v2[1] - v2[0] * v1[1]]
    return unit_vector(normal)


def get_angle(v1, v2, units='rads'):
    """
    Calculate the angle between two vectors.

    Args:
         v1 (list): Vector 1.
         v2 (list): Vector 2.
         units (str, optional): The desired units, either 'rads' for radians, or 'degs' for degrees.

    Returns:
        float: The angle between ``v1`` and ``v2`` in the desired units.

    Raises:
        VectorsError: If ``v1`` and ``v2`` are of different lengths.
    """
    if len(v1) != len(v2):
        raise VectorsError(f'v1 and v2 must be the same length, got {len(v1)} and {len(v2)}.')
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    conversion = 180 / math.pi if 'degs' in units else 1
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * conversion


def get_dihedral(v1, v2, v3, units='degs'):
    """
    Calculate the dihedral angle between three vectors.
    ``v2`` connects between ``v1`` and ``v3``.
    Inspired by ASE Atoms.get_dihedral().

    Args:
         v1 (list): Vector 1.
         v2 (list): Vector 2.
         v3 (list): Vector 3.
         units (str, optional): The desired units, either 'rads' for radians, or 'degs' for degrees.

    Returns:
        float: The dihedral angle between ``v1`` and ``v2`` in the desired units.

    Raises:
        VectorsError: If either ``v1`` or ``v2`` have lengths different than three.
    """
    if len(v1) != 3 or len(v2) != 3 or len(v3) != 3:
        raise VectorsError(f'v1, v2, and v3 must have a length of three, got {len(v1)}, {len(v2)}, and {len(v3)}.')
    v1, v2, v3 = np.array(v1, np.float64), np.array(v2, np.float64), np.array(v3, np.float64)
    v2_x_v1 = np.cross(v2, v1)
    v2_x_v1 /= float(np.linalg.norm(v2_x_v1))
    v3_x_v2 = np.cross(v3, v2)
    v3_x_v2 /= np.linalg.norm(v3_x_v2)
    dihedral = np.arccos(np.clip(np.vdot(v2_x_v1, v3_x_v2), -1, 1))
    if np.vdot(v2_x_v1, v3) > 0:
        dihedral = 2 * np.pi - dihedral
    if np.isnan(dihedral):
        raise VectorsError('Could not calculate a dihedral angle')
    conversion = 180 / math.pi if 'degs' in units else 1
    return dihedral * conversion


def calculate_distance(coords, atoms, index=0):
    """
    Calculate an angle.

    Args:
        coords (list, tuple): The array-format or tuple-format coordinates.
        atoms (list): The 2 atoms to calculate defining the vector for which the length will be calculated.
        index (int, optional): Whether ``torsion`` is 0-indexed or 1-indexed (values are 0 or 1).

    Returns:
        float: The distance in the coords units.

    Raises:
        VectorsError: If ``index`` is out of range, or ``atoms`` is of wrong length or has repeating indices.
    """
    if index not in [0, 1]:
        raise VectorsError(f'index must be either 0 or 1, got {index}')
    if len(atoms) != 2:
        raise VectorsError(f'distance atom list must be of length two, got {len(atoms)}')
    if len(set(atoms)) < 2:
        raise VectorsError(f'some atoms are repetitive: {atoms}')
    atoms = [a - index for a in atoms]  # convert 1-index to 0-index
    coords = np.asarray(coords, dtype=np.float32)
    vector = coords[atoms[1]] - coords[atoms[0]]
    return get_vector_length(vector)


def calculate_angle(coords, atoms, index=0, units='degs'):
    """
    Calculate an angle.

    Args:
        coords (list, tuple): The array-format or tuple-format coordinates.
        atoms (list): The 3 atoms defining the angle.
        index (int, optional): Whether ``torsion`` is 0-indexed or 1-indexed (values are 0 or 1).
        units (str, optional): The desired units, either 'rads' for radians, or 'degs' for degrees.

    Returns:
        float: The angle.

    Raises:
        VectorsError: If ``index`` is out of range, or ``atoms`` is of wrong length or has repeating indices.
    """
    if index not in [0, 1]:
        raise VectorsError(f'index must be either 0 or 1, got {index}')
    if len(atoms) != 3:
        raise VectorsError(f'angle atom list must be of length three, got {len(atoms)}')
    if len(set(atoms)) < 3:
        raise VectorsError(f'some atoms are repetitive: {atoms}')
    atoms = [a - index for a in atoms]  # convert 1-index to 0-index
    coords = np.asarray(coords, dtype=np.float32)
    v1 = coords[atoms[1]] - coords[atoms[0]]
    v2 = coords[atoms[1]] - coords[atoms[2]]
    return get_angle(v1, v2, units=units)


def calculate_dihedral_angle(coords, torsion, index=0, units='degs'):
    """
    Calculate a dihedral angle.

    Args:
        coords (list, tuple): The array-format or tuple-format coordinates.
        torsion (list): The 4 atoms defining the dihedral angle.
        index (int, optional): Whether ``torsion`` is 0-indexed or 1-indexed (values are 0 or 1).
        units (str, optional): The desired units, either 'rads' for radians, or 'degs' for degrees.

    Returns:
        float: The dihedral angle.

    Raises:
        VectorsError: If ``index`` is out of range, or ``torsion`` is of wrong length or has repeating indices.
    """
    if index not in [0, 1]:
        raise VectorsError(f'index must be either 0 or 1, got {index}')
    if len(torsion) != 4:
        raise VectorsError(f'torsion atom list must be of length four, got {len(torsion)}')
    if len(set(torsion)) < 4:
        raise VectorsError(f'some indices in torsion are repetitive: {torsion}')
    torsion = [t - index for t in torsion]  # convert 1-index to 0-index if needed
    coords = np.asarray(coords, dtype=np.float32)
    v1 = coords[torsion[1]] - coords[torsion[0]]
    v2 = coords[torsion[2]] - coords[torsion[1]]
    v3 = coords[torsion[3]] - coords[torsion[2]]
    return get_dihedral(v1, v2, v3, units=units)


def unit_vector(vector):
    """
    Calculate a unit vector in the same direction as the input vector.

    Args:
        vector (list): The input vector.

    Returns:
        list: The unit vector.
    """
    length = get_vector_length(vector)
    return [vi / length for vi in vector]


def set_vector_length(vector, length):
    """
    Set the length of a 3D vector.

    Args:
        vector (list): The vector to process.
        length (float): The desired length to set.

    Returns:
        list: A vector with the desired length.
    """
    u = unit_vector(vector)
    return [u[0] * length, u[1] * length, u[2] * length]


def rotate_vector(point_a, point_b, normal, theta):
    """
    Rotate a vector in 3D space around a given axis by a certain angle.

    Inspired by https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    Args:
        point_a (list): The 3D coordinates of the starting point (point A) of the vector to be rotated.
        point_b (list): The 3D coordinates of the ending point (point B) of the vector to be rotated.
        normal (list): The axis to be rotated around.
        theta (float): The degree in radians by which to rotate.

    Returns:
        list: The rotated vector (the new coordinates for point B).
    """
    normal = np.asarray(normal)
    normal = normal / math.sqrt(np.dot(normal, normal))  # should *not* be replaced by an augmented assignment
    a = math.cos(theta / 2.0)
    b, c, d = -normal * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    vector = [b - a for b, a in zip(point_b, point_a)]
    new_vector = np.dot(rotation_matrix, vector).tolist()
    new_vector = [n_v + a for n_v, a in zip(new_vector, point_a)]  # root the vector at the original starting point
    return new_vector


def get_vector(pivot, anchor, xyz):
    """
    Get a vector between two atoms in the molecule (pointing from pivot to anchor).

    Args:
        pivot (int): The 0-index of the pivotal atom around which groups are to be translated.
        anchor (int): The 0-index of an additional atom in the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.

    Returns:
         list: A vector pointing from the pivotal atom towards the anchor atom.
    """
    x, y, z = converter.xyz_to_x_y_z(xyz)
    dx = x[anchor] - x[pivot]
    dy = y[anchor] - y[pivot]
    dz = z[anchor] - z[pivot]
    return [dx, dy, dz]


def get_lp_vector(label, mol, xyz, pivot):
    """
    Get a vector from the pivotal atom in the molecule towards its lone electron pair (lp).
    The approach is to reverse the average of the three unit vectors between the pivotal atom and its neighbors.

    Args:
        label (str): The species' label.
        mol (Molecule): The 2D graph representation of the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.
        pivot (int): The 0-index of the pivotal atom of interest.

    Returns:
        list: A unit vector pointing from the pivotal (nitrogen) atom towards its lone electron pairs orbital.

    Raises:
        VectorsError: If the lp vector cannot be attained.
    """
    neighbors, vectors = list(), list()
    if not mol.atoms[pivot].is_nitrogen():
        logger.warning(f'The get_lp_vector specializes in nitrogen atoms, got {mol.atoms[pivot].symbol} '
                       f'(atom number {pivot}) in species {label}.')
    for atom in mol.atoms[pivot].edges.keys():
        neighbors.append(mol.atoms.index(atom))
    if len(neighbors) < 3:
        # N will have 3, S may have more.
        raise VectorsError(f'Can only get lp vector if the pivotal atom has at least three neighbors. '
                           f'Atom {mol.atoms[pivot]} in {label} has only {len(neighbors)}.')
    for neighbor in neighbors:
        # already taken in "reverse", pointing towards the lone pair
        vectors.append(unit_vector(get_vector(pivot=neighbor, anchor=pivot, xyz=xyz)))
    x = sum(v[0] for v in vectors)
    y = sum(v[1] for v in vectors)
    z = sum(v[2] for v in vectors)
    return unit_vector([x, y, z])


def get_vector_length(v):
    """
    Get the length of an ND vector

    Args:
        v (list): The vector.

    Returns:
        float: The vector's length.
    """
    return np.dot(v, v) ** 0.5
