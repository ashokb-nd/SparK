"""
It does two things
1. sends the tail of a std_output backup file as a teams notification on regular intervals.
2. logs the 'log_dir' to mlflow.

"""

import requests
import json
import os
import time
import mlflow
import argparse
import logging
logger = logging.getLogger(__name__)

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
                logger.info("Message sent successfully!")
            else:
                logger.error(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def close_session(self):
        self.session.close()  # Close the session when done

def read_tail(filepath,rows=10):
    cmd = f'tail -n {rows} {filepath}'
    output = os.popen(cmd).read()
    return output

if __name__ == "__main__":

    ASHOK_WEBHOOK_URL =  "https://netorg726775.webhook.office.com/webhookb2/8cbcc9af-7602-4c4f-b5b1-d46ba143455e@b84f219a-0fcd-4dfa-8476-edcc96f3324c/IncomingWebhook/f351da18983f4fe3ad0959620e93569e/4353d6e9-f97d-47b6-8012-461de8807c18/V2rMGo_swhGuYoHT-XIHuEJfC6CRWd-wZiRcs6Ty0OUlk1" 

    parser = argparse.ArgumentParser(description='Send messages to Microsoft Teams.')

    parser.add_argument('--webhook_url', type=str, default=ASHOK_WEBHOOK_URL, help='Microsoft Teams webhook URL')
    parser.add_argument("--tag", type=str, default="spark", help="Tag for the experiment")
    parser.add_argument("--filepath", type=str, default="/ashok/SparK/logdir/stdout_backup.txt", help="Path to Read the loss values")
    parser.add_argument("--experiment_dir", type=str, default="/ashok/SparK/logdir/", help="Directory to log the artifacts")
    parser.add_argument("--sleep",type=int, default=3600, help="sleep time between messages")
    parser.add_argument("--log_freq",type=int, default=4, help="logs the experiment_dir once every `log_freq` times of messaging.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')


    messenger = TeamsMessenger(args.webhook_url)

    i = 0
    while True:

        #every hour send last 4 lines of the log file
        message = read_tail(args.filepath, 4)
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        header = args.tag +" @ " + datetime + "\n"
        messenger.send_message(header + message)
# 
        logger.debug(header + message)

        if (i)%args.log_freq == 0: # every 4 hours 
            logger.info("Logging artifacts to MLflow...")
            mlflow.log_artifact(args.experiment_dir)

        logger.debug(f"Sleeping for {args.sleep} seconds...")

        time.sleep(args.sleep)
        i += 1

    messenger.close_session()
