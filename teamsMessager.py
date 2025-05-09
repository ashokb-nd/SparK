import requests
import json
import os
import time
import mlflow

class TeamsMessenger:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.session = requests.Session()  # Create a session object

    def send_message(self, message_text):
        message = {"text": message_text}
        try:
            response = self.session.post(  # Use the session to send the request
                self.webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(message)
            )
            if response.status_code == 200:
                print("Message sent successfully!")
            else:
                print(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def close_session(self):
        self.session.close()  # Close the session when done

# Example usage:
# webhook_url = "https://your-webhook-url"
# messenger = TeamsMessenger(webhook_url)
# messenger.send_message("Hello, this is a test message sent from Python to Microsoft Teams!")

def read_tail(filepath,rows=10):
    cmd = f'tail -n {rows} {filepath}'
    output = os.popen(cmd).read()
    return output

if __name__ == "__main__":
    webhook_url = "https://netorg726775.webhook.office.com/webhookb2/8cbcc9af-7602-4c4f-b5b1-d46ba143455e@b84f219a-0fcd-4dfa-8476-edcc96f3324c/IncomingWebhook/f351da18983f4fe3ad0959620e93569e/4353d6e9-f97d-47b6-8012-461de8807c18/V2rMGo_swhGuYoHT-XIHuEJfC6CRWd-wZiRcs6Ty0OUlk1"
    messenger = TeamsMessenger(webhook_url)

    filepath =  "/ashok/SparK/logdir/stdout_backup.txt"
    experiment_dir = '/ashok/SparK/logdir/', # to log the artifacts

    i = 0
    while True:

        #every hour send last 4 lines of the log file
        message = read_tail(filepath, 4)
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        header = "spark (Imagenet pretrained) @ " + datetime + "\n"
        messenger.send_message(header + message)
        print(header + message)
        time.sleep(3600)

        if (i+1)%4 == 0:
            print("Logging artifacts to MLflow")
            mlflow.artifacts.logartifact(experiment_dir, artifact_path="experiment_logs")


    messenger.close_session()
