# tests/test_app.py
import unittest
from app import create_app

class BasicTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        response = self.app.post('/predict', data=dict(Bedrooms=3, Bathrooms=2, Area=1500, Location='Suburb'))
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
