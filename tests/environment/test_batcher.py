from OfflineOnline.environment import VguiBatch

import asyncio

async def main():
    env = await VguiBatch.create(4)
    # env.start()
    env.end()

asyncio.run(main())