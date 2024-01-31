from tqdm import tqdm
import openai
import time
import os

class ContentModeration:
    def __init__(self, open_ai_key):
        openai.api_key = open_ai_key #os.environ["OPEN_AI_KEY"] #open_ai_key #os.environ["OPEN_AI_KEY"]
        # self.cm = ContentModeration()
    def get_hate(self, message):
        """
        Run content moderation on a single message
        :param message:
        :return:
        """
        response = openai.Moderation.create(
            input=message,
        )
        if 'results' in response and response['results']:
            return response['results'][0]["category_scores"]
        else:
            return None

    def content_moderation(self, messages):
        """
        Run content moderation on a list of messages
        :param messages:
        :return:
        """
        collect = []
        for o in tqdm(messages, total=len(messages)):
            collect.append(self.get_hate(o))
            time.sleep(0.7)

        return collect