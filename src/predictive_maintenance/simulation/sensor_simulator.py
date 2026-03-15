import random
import time
import requests


API_URL = "http://127.0.0.1:8000/predict"


def generate_sensor_data():

    return {
        "air_temperature": random.uniform(295, 305),
        "process_temperature": random.uniform(305, 315),
        "rotational_speed": random.uniform(1200, 1800),
        "torque": random.uniform(30, 50),
        "tool_wear": random.uniform(0, 200),
    }


def run_simulation():

    while True:

        data = generate_sensor_data()

        response = requests.post(API_URL, json=data)

        print(response.json())

        time.sleep(2)


if __name__ == "__main__":
    run_simulation()