from django.test import TestCase

# Create your tests here.
from django.test import TestCase, Client
from django.urls import reverse

class APITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_predict_endpoint(self):
        response = self.client.post(reverse('weather_prediction'), {
            "gu_name": "Gangseo-gu",
            "dong_name": "Gonghang-dong"
        }, content_type='application/json')
        
        print(response.content)  # 응답 내용을 출력하여 디버깅
        self.assertEqual(response.status_code, 200)
