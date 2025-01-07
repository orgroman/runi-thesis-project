from aiohttp import ClientSession

class LLamaFileModel:
    def __init__(self, **kwargs):
        pass
                
    async def apredict_text(self, text):
        async with ClientSession() as session:
            async with session.get(self.url, params={'text': text}) as response:
                return bool(await response.json()['result'])
