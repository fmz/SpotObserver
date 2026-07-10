
import cv2
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple

from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b


def correspondence(
        source_points: np.ndarray, target_points: np.ndarray, max_distance: float = 0.1
        ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Find correspondences between source and target points using nearest neighbor search.

    Args:
        source_points (np.ndarray): Source points of shape (N, 3).
        target_points (np.ndarray): Target points of shape (M, 3).
        max_distance (float): Maximum distance to consider a valid correspondence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices of corresponding points in source and target.
    """

    # nearest neighbor search
    tree = cKDTree(target_points)
    distances, indices = tree.query(source_points)

    # filtering correspondences based on max_distance
    valid_mask = distances < max_distance
    return np.where(valid_mask)[0], indices[valid_mask]

def kabsch(
        source_points: np.ndarray, target_points: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Estimate the rigid transformation (rotation and translation) that 
    aligns source points to target points.
    
    Args:
        source_points (np.ndarray): Source points of shape (N, 3).
        target_points (np.ndarray): Target points of shape (N, 3).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix (3x3) and translation vector.
    """

    # compute centroids of both point sets
    p = np.mean(source_points, axis = 0)
    q = np.mean(target_points, axis = 0)

    # center the points around their centroids
    source_centered = source_points - p
    target_centered = target_points - q

    # compute the covariance matrix
    covariance_matrix = source_centered.T @ target_centered
    u_mat, _, v_mat = np.linalg.svd(covariance_matrix)

    # compute the rotation matrix using the kabsch algorithm
    rotation_matrix = v_mat.T @ u_mat.T

    # ensure a proper rotation (determinant = 1)
    if np.linalg.det(rotation_matrix) < 0:
        v_mat[-1, :] *= -1
        rotation_matrix = v_mat.T @ u_mat.T

    # compute the translation vector by aligning the centroids
    translation_vector = q - rotation_matrix @ p

    return rotation_matrix, translation_vector

def compute_initial_positions(pose_a, pose_b):
    """
    Compute the rotation and translation that maps robot A's body frame into
    robot B's body frame, for use as icp()'s initial_rotation/initial_translation.

    IMPORTANT: pose_a and pose_b must be expressed in a genuinely shared
    reference frame (e.g. both robots localized against the same GraphNav map,
    via seed_tform_body) for this to be a meaningful cross-robot estimate. If
    they are each robot's own independent vision_tform_body with no shared
    anchor between the two robots, this still returns a number, but it is not
    a verified relationship -- it is no more trustworthy than identity until a
    real shared frame is in place.

    Args:
        pose_a (np.ndarray): 4x4 transformation matrix from robot A's body frame to the shared frame.
        pose_b (np.ndarray): 4x4 transformation matrix from robot B's body frame to the shared frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix (3x3) and translation vector
        mapping robot A's body-frame points into robot B's body frame.
    """

    body_b_tform_body_a = np.linalg.inv(pose_b) @ pose_a

    rotation = body_b_tform_body_a[:3, :3]
    translation = body_b_tform_body_a[:3, 3]

    return rotation, translation

def icp(
        source_points: np.ndarray, target_points: np.ndarray, 
        max_iterations: int = 100, tolerance: float = 1e-6, max_distance: float = 0.1,
        initial_rotation: np.ndarray = None, initial_translation: np.ndarray = None
        ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Perform Iterative Closest Point (ICP) algorithm to align source points to target points.

    Args:
        source_points (np.ndarray): Source points of shape (N, 3).
        target_points (np.ndarray): Target points of shape (M, 3).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        max_distance (float): Maximum distance to consider a valid correspondence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Final rotation matrix and translation vector.
    """

    # initialize transformation to return at the end
    rotation_total = np.eye(3) if initial_rotation is None else np.array(initial_rotation, dtype = float)
    translation_total = np.zeros(3) if initial_translation is None else np.array(initial_translation, dtype = float)

    # iterate to refine transform
    for i in range(max_iterations):

        # apply current transformation to source points
        source_transformed = (rotation_total @ source_points.T).T + translation_total
        source_indices, target_indices = correspondence(source_transformed, target_points, max_distance)

        if len(source_indices) < 3:
            print( "Not enough correspondences found." )
            break
        
        # compute the optimal rotation and translation using kabsch algorithm
        rotation_vec, translation_vec = kabsch(source_transformed[source_indices], target_points[target_indices])
        
        rotation_total = rotation_vec @ rotation_total
        translation_total = rotation_vec @ translation_total + translation_vec

        if np.linalg.norm(translation_vec) < tolerance and np.linalg.norm(rotation_vec - np.eye(3)) < tolerance:
            print(f"Converged after {i + 1} iterations.")
            break
    
    return rotation_total, translation_total