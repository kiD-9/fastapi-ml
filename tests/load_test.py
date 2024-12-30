import io

from locust import HttpUser, between, task
from PIL import Image


class PredictionUser(HttpUser):
    wait_time = between(1, 5)  # Время ожидания между запросами

    @task
    def predict(self):
        # Загрузите изображение
        image_path = 'img.jpg'
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Отправьте POST-запрос с изображением
        response = self.client.post("/predict", files={"file": ("image.jpg", io.BytesIO(image_data), "image/jpeg")})

        # Проверьте ответ
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print("Failed:", response.status_code, response.text)