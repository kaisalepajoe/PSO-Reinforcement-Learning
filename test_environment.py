import unittest
import numpy as np

from environment import get_limb_end


class GetLimbEndTests(unittest.TestCase):
	def test_an_angle(self):
		x1, y1 = get_limb_end(0,0,5,np.pi/6)
		#print(x1, y1)
		self.assertTrue(np.isclose(x1, 2.5))
		self.assertTrue(np.isclose(y1, 4.330127019))
	def test_negative_angle(self):
		x1, y1 = get_limb_end(0,0,5,-np.pi/6)
		#print(x1, y1)
		self.assertTrue(np.isclose(x1, -2.5))
		self.assertTrue(np.isclose(y1, 4.330127019))
	def test_positive_starting_point(self):
		x1, y1 = get_limb_end(1,2,5,np.pi/6)
		#print(x1, y1)
		self.assertTrue(np.isclose(x1, 3.5))
		self.assertTrue(np.isclose(y1, 6.330127019))
	def test_negative_starting_point(self):
		x1, y1 = get_limb_end(-2,-1,5,np.pi/6)
		#print(x1, y1)
		self.assertTrue(np.isclose(x1, 0.5))
		self.assertTrue(np.isclose(y1, 3.330127019))

if __name__ == '__main__':
	unittest.main()