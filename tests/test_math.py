import unittest

import numpy as np

from visualizer.matrix_transforms import identify_matrix_type, rodrigues_rotation_matrix
from visualizer.plotting_utils import (
    add_cube_mesh,
    add_grid_3d,
    add_sphere_mesh,
    add_vector_3d,
    create_base_figure,
    matrix_to_latex,
)


class TestMatrixVisualizer(unittest.TestCase):
    def test_rodrigues_rotation(self):
        axis = [0, 0, 1]
        theta = np.pi / 2
        R = rodrigues_rotation_matrix(axis, theta)

        # R powinno być macierzą ortogonalną (R @ R.T == I) oraz det(R) == 1
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        self.assertTrue(np.isclose(np.linalg.det(R), 1.0, atol=1e-6))

        # Kolumny macierzy R to nowe wektory jednostkowe
        col_x = R[:, 0]
        expected_col_x = np.array([0.0, 1.0, 0.0])
        self.assertTrue(np.allclose(col_x, expected_col_x, atol=1e-6))

    def test_matrix_multiplication_properties(self):
        A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        B = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

        # det(A @ B) == det(A) * det(B)
        det_AB = np.linalg.det(A @ B)
        det_A_det_B = np.linalg.det(A) * np.linalg.det(B)
        self.assertTrue(np.isclose(det_AB, det_A_det_B, atol=1e-5))

        # Nieprzemienność: A @ B != B @ A
        self.assertFalse(np.allclose(A @ B, B @ A))

    def test_matrix_inverse(self):
        A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 1.0]])
        self.assertGreater(abs(np.linalg.det(A)), 1e-4)

        A_inv = np.linalg.inv(A)
        self.assertTrue(np.allclose(A @ A_inv, np.eye(3), atol=1e-6))

    def test_matrix_eigen(self):
        A = np.diag([3.0, -1.0, 2.0])
        eigenvalues, eigenvectors = np.linalg.eig(A)

        self.assertEqual(set(np.round(eigenvalues, 4)), {3.0, -1.0, 2.0})

        # A @ v == lambda * v
        for i in range(3):
            lam = eigenvalues[i]
            v = eigenvectors[:, i]
            self.assertTrue(np.allclose(A @ v, lam * v, atol=1e-6))

    def test_identify_matrix_type(self):
        I = np.eye(3)
        types_I = identify_matrix_type(I)
        self.assertTrue(any("Czysty obrót" in t or "Ortogonalna" in t for t in types_I))

        singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        types_sing = identify_matrix_type(singular)
        self.assertTrue(any("Osobliwa" in t for t in types_sing))

    def test_plotting_helpers(self):
        fig = create_base_figure(title="Test", limit=3.0)
        add_vector_3d(fig, [1, 2, 3], name="kolumna")
        add_cube_mesh(fig, transform_matrix=np.eye(3))
        add_sphere_mesh(fig, transform_matrix=np.eye(3))
        add_grid_3d(fig, transform_matrix=np.eye(3))

        self.assertGreater(len(fig.data), 0)

    def test_latex_formatting(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        latex_str = matrix_to_latex(M)
        self.assertIn(r"\begin{bmatrix}", latex_str)
        self.assertIn(r"\end{bmatrix}", latex_str)


if __name__ == "__main__":
    unittest.main()
