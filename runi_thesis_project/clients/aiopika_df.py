import asyncio
import json
import pandas as pd
import aio_pika
from aio_pika.abc import AbstractChannel, AbstractQueue
import zlib

def compress_body(data: bytes, level: int = 6) -> bytes:
    """
    Compress byte data using zlib.
    :param data: The raw bytes to compress.
    :param level: Compression level (1-9). Default is 6 for a balance of speed/size.
    :return: Compressed bytes.
    """
    return zlib.compress(data, level)

def decompress_body(data: bytes) -> bytes:
    """
    Decompress zlib-compressed byte data.
    :param data: The compressed bytes.
    :return: Decompressed (original) bytes.
    """
    return zlib.decompress(data)

class AioPikaDataFrameClient:
    """
    A client that:
      - Connects robustly to RabbitMQ via aio-pika.
      - Publishes a Pandas DataFrame row-by-row (each row serialized as dict->JSON) to an input queue.
      - Consumes from the input queue, transforms each row, and re-publishes to a results queue.
      - Consumes from the results queue to build a final DataFrame in memory.

    Now with optional compression support.
    """

    def __init__(
        self,
        amqp_url: str = "amqp://guest:guest@127.0.0.1/",
        input_queue_name: str = "df_queue",
        results_queue_name: str = "df_results_queue",
        compression_enabled: bool = True,
        compression_level: int = 6,
    ):
        """
        :param amqp_url: Connection string to RabbitMQ.
        :param input_queue_name: The name of the queue for DF messages to be transformed.
        :param results_queue_name: The name of the queue for transformed DF messages.
        :param compression_enabled: Whether to compress the JSON payload before publishing.
        :param compression_level: Compression level (1-9) if compression is enabled.
        """
        self.amqp_url = amqp_url
        self.input_queue_name = input_queue_name
        self.results_queue_name = results_queue_name

        self.connection: aio_pika.RobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.input_queue: AbstractQueue | None = None
        self.results_queue: AbstractQueue | None = None

        self.compression_enabled = compression_enabled
        self.compression_level = compression_level

    async def connect(self) -> None:
        """
        Establish a robust connection and declare both the input and results queues.
        """
        self.connection = await aio_pika.connect_robust(self.amqp_url)
        # Create channel with publisher confirms to ensure reliability
        self.channel = await self.connection.channel(publisher_confirms=True)
        await self.channel.set_qos(prefetch_count=5)

        # Declare the input queue (from which we consume)
        self.input_queue = await self.channel.declare_queue(
            self.input_queue_name, durable=True
        )

        # Declare the results queue (to which we publish)
        self.results_queue = await self.channel.declare_queue(
            self.results_queue_name, durable=True
        )

    async def close(self) -> None:
        """
        Gracefully close channel and connection.
        """
        if self.channel is not None:
            await self.channel.close()
        if self.connection is not None:
            await self.connection.close()

    #
    # 1) Publish a DataFrame to the input queue
    #
    async def publish_dataframe(self, df: pd.DataFrame) -> None:
        """
        Publish a DataFrame row-by-row to the 'input' queue.
        """
        if not self.channel:
            raise RuntimeError("Call connect() before publishing.")

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            body = json.dumps(row_dict).encode("utf-8")

            # Compress if enabled
            if self.compression_enabled:
                body = compress_body(body, level=self.compression_level)

            message = aio_pika.Message(
                body=body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            await self.channel.default_exchange.publish(
                message=message,
                routing_key=self.input_queue_name,
            )

    #
    # 2) Consume from input queue -> transform -> publish to results queue
    #
    async def transform_row_and_republish(self, row_dict: dict, transform_callback):
        """
        Transform the row via transform_callback, then publish to the results queue.
        """
        # If transform_callback is truly async, we must await it
        # If it's sync, calling 'await' on it won't work, so you'll need to remove 'await' if it's purely sync.
        transformed_dict = await transform_callback(row_dict)

        body = json.dumps(transformed_dict).encode("utf-8")

        # Compress if enabled
        if self.compression_enabled:
            body = compress_body(body, level=self.compression_level)

        msg = aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        await self.channel.default_exchange.publish(
            msg,
            routing_key=self.results_queue_name
        )

    async def consume_and_transform(
        self,
        transform_callback,
        expected_count: int | None = None
    ):
        """
        Consume messages from the input queue, transform them,
        and re-publish them to the results queue.
        
        :param transform_callback: An async or sync callable that transforms each row dict.
        :param expected_count: Optionally stop after consuming this many messages.
        """
        if not self.input_queue:
            raise RuntimeError("Call connect() before consuming.")

        count = 0

        async with self.input_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    # Decompress if needed
                    raw_body = message.body
                    if self.compression_enabled:
                        raw_body = decompress_body(raw_body)

                    row_dict = json.loads(raw_body)
                    await self.transform_row_and_republish(
                        row_dict, transform_callback
                    )

                count += 1
                if expected_count is not None and count >= expected_count:
                    # Stop reading once we have consumed enough
                    await queue_iter.close()
                    break

    #
    # 3) Consume from results queue -> accumulate in DataFrame
    #
    async def consume_results(
        self,
        expected_count: int | None = None
    ) -> pd.DataFrame:
        """
        Consume messages from the results queue and accumulate them into a DataFrame.
        
        :param expected_count: If known, the number of messages to consume.
        :return: A pandas DataFrame containing the consumed rows.
        """
        if not self.results_queue:
            raise RuntimeError("Call connect() before consuming results.")

        results = []
        count = 0

        async with self.results_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    # Decompress if needed
                    raw_body = message.body
                    if self.compression_enabled:
                        raw_body = decompress_body(raw_body)

                    row_dict = json.loads(raw_body)
                    results.append(row_dict)

                count += 1
                if expected_count is not None and count >= expected_count:
                    await queue_iter.close()
                    break

        return pd.DataFrame(results)

#
# Example transform callback
#
async def transform_example(row: dict) -> dict:
    """
    A sample transform function that sums numeric fields in the row and sets 'sum_col'.
    
    NOTE: If you do NOT need async, define this as a normal function (def transform_example(...))
    and remove the 'await' in transform_row_and_republish.
    """
    numeric_sum = sum(v for v in row.values() if isinstance(v, (int, float)))
    row["sum_col"] = numeric_sum
    return row

#
# Bringing it all together
#
async def main():
    original_df = pd.DataFrame([
        {"A": 1, "B": 2},
        {"A": 10, "B": 20},
        {"A": 100, "B": 200},
    ])

    client = AioPikaDataFrameClient(
        amqp_url="amqp://admin:password@127.0.0.1/",
        input_queue_name="df_queue",
        results_queue_name="df_results_queue",
        compression_enabled=True,      # Enable or disable compression
        compression_level=6            # Adjust compression level as desired
    )

    # 1. Connect and publish the original DataFrame
    await client.connect()
    print("Publishing DataFrame rows to input queue...")
    await client.publish_dataframe(original_df)
    print("Done publishing.")

    # 2. Consume from input queue, transform, and re-publish to results queue
    #    We'll consume exactly len(original_df) messages
    print("Starting consumer to transform and re-publish...")
    await client.consume_and_transform(
        transform_callback=transform_example,
        expected_count=len(original_df)
    )
    print("Done consuming and transforming to results queue.")

    # 3. Finally, consume from results queue to build the final DataFrame
    print("Consuming transformed rows from results queue...")
    transformed_df = await client.consume_results(
        expected_count=len(original_df)
    )
    print("Done consuming from results queue.\nTransformed DataFrame:")
    print(transformed_df)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
