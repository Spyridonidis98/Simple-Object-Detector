import unittest
class TestIOU(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.bboxes_true = [
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.2, 0.2],
            [0.95, 0.7, 0.3, 0.2],
            [0.7, 0.95, 0.6, 0.1],
            ]

        self.bboxes_pred = [
            [0.5, 0.5, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.9, 0.2, 0.2, 0.2],
            [0.95, 0.6, 0.5, 0.2],
            [0.5, 1.15, 0.4, 0.7],
            ]
        
        self.true_iou = [0.25/1.75, 0.25/1.75, 0.5/1.5, 0.5/1.5, 1.0/4.0, 1.0/4.0, 0.25/4.75, 0.25/4.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25/1.75, 0.75/3.25, 3/31]

        self.bboxes_true = np.array(self.bboxes_true)
        self.bboxes_pred = np.array(self.bboxes_pred)
        self.true_iou = np.array(self.true_iou)

    def test_equal(self):
        import sys
        sys.path.append('../')
        from sodv1.utils import IOU

        pred_iou = IOU(self.bboxes_true, self.bboxes_pred)
        is_equal = abs(pred_iou-self.true_iou)<1e-5  
        for i in range(len(pred_iou)):
            self.assertEqual(is_equal[i], True, f"ious with id = {i} are not equal")

if __name__ == "__main__":
    unittest.main()