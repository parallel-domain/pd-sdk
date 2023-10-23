import unittest

import numpy as np

from paralleldomain.model.geometry.bezier_curve_2d import BezierCurve2DGeometry
from paralleldomain.model.geometry.point_2d import Point2DGeometry


class TestBezierCurve2DBaseGeometry(unittest.TestCase):
    def setUp(self):
        self.control_points = [Point2DGeometry(0, 0), Point2DGeometry(1, 1), Point2DGeometry(2, 0)]
        self.bezier = BezierCurve2DGeometry(self.control_points)

    def test_length(self):
        self.assertAlmostEqual(self.bezier.length, 2.295, places=2)

    def test_control_points(self):
        np.testing.assert_almost_equal(self.bezier.control_points, np.array([[0, 0], [1, 1], [2, 0]]))

    def test_evaluate_at(self):
        point = self.bezier.evaluate_at(0.5)
        self.assertAlmostEqual(point.x, 1.0, places=2)
        self.assertAlmostEqual(point.y, 0.5, places=2)

    def test_tangent_at(self):
        tangent = self.bezier.tangent_at(0.5)
        self.assertAlmostEqual(tangent.x, 1.0, places=2)
        self.assertAlmostEqual(tangent.y, 0.0, places=2)

    def test_as_polyline(self):
        polyline = self.bezier.as_polyline(num_points=3)
        self.assertEqual(len(polyline.lines), 2)
        self.assertAlmostEqual(polyline.lines[0].start.x, 0.0, places=2)
        self.assertAlmostEqual(polyline.lines[0].start.y, 0.0, places=2)
        self.assertAlmostEqual(polyline.lines[1].start.x, 1.0, places=2)
        self.assertAlmostEqual(polyline.lines[1].start.y, 0.5, places=2)
        self.assertAlmostEqual(polyline.lines[1].end.x, 2.0, places=2)
        self.assertAlmostEqual(polyline.lines[1].end.y, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
