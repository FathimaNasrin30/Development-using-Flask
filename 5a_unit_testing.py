import unittest
from EmotionDetection import emotion_detector

class TestEmotionDetection(unittest.TestCase):

    def test_joy(self):
        """Test case for joy emotion"""
        result = emotion_detector("I am very happy today!")
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_anger(self):
        """Test case for anger emotion"""
        result = emotion_detector("I am really mad at this!")
        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_sadness(self):
        """Test case for sadness emotion"""
        result = emotion_detector("I feel very sad")
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_empty_input(self):
        """Test case for empty input"""
        result = emotion_detector("")
        self.assertIsNone(result['dominant_emotion'])

if __name__ == "__main__":
    unittest.main()
