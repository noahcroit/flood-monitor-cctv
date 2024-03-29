import asyncio
import aioredis
import async_timeout
import queue
import argparse
import json
from pymodbus.client import ModbusTcpClient



tagname_floodgate_up   = "watergate.floodgate.gate-up"
tagname_floodgate_down = "watergate.floodgate.gate-down"
tagname_floodgate_pos  = "watergate.floodgate.gate-position-get"



async def redis_sub(channel: aioredis.client.PubSub, q_write_msg):
    while True:
        try:
            async with async_timeout.timeout(1):
                message = await channel.get_message(ignore_subscribe_messages=True)
                if message is not None:
                    print(f"(Reader) Message Received: {message}")
                    ch_name = message['channel'].decode('utf-8')
                    data = int(message['data'])
                    q_write_msg.put([ch_name, data])
                await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            pass

async def redis_pub(redis, q_read_msg):
    while True:
        if not q_read_msg.empty():
            val = q_read_msg.get()
            tagname  = val[0]
            read_val = val[1]
            await redis.set("tag:"+tagname, str(read_val))
        else:
            print("pub Q is empty..")
        await asyncio.sleep(0.5)

async def modbus_worker(q_write_msg, q_read_msg):
    global tagname_floodgate_up
    global tagname_floodgate_down
    global tagname_floodgate_pos

    # Modbus Device initialize
    print("Modbus Initialize")
    client = ModbusTcpClient('192.168.0.50')
    client.connect()
    await asyncio.sleep(3)

    while True:
        # Write to device section
        if not q_write_msg.empty():
            [ch, val] = q_write_msg.get()
            ch = ch.strip('settag:')
            await asyncio.sleep(1)
            try:
                if ch == tagname_floodgate_up:
                    # write MODBUS, multi-coil
                    print("Write MODBUS!!!!!!!!!!!!!!!!!, command=UP")
                    if not val == 0:
                        client.write_coils(address=160, values=[1, 0], slave=0)
                        await asyncio.sleep(1)
                        client.write_coils(address=160, values=[0, 0], slave=0)
                elif ch == tagname_floodgate_down:
                    # write MODBUS, multi-coil
                    print("Write MODBUS!!!!!!!!!!!!!!!!!, command=DOWN")
                    if not val == 0:
                        client.write_coils(address=320, values=[1, 0], slave=0)
                        await asyncio.sleep(1)
                        client.write_coils(address=320, values=[0, 0], slave=0)
            except:
                print("Write MODBUS fail!")
                await asyncio.sleep(5)

        # Read from device section
        try:
            print("Read coil for floodgate up")
            #response = client.read_coils(address=160, count=8, slave=0)
            #value = response.registers
            #q_read_msg.put([tagname_floodgate_up, value])
            #await asyncio.sleep(0.2)

            print("Read coil for floodgate down")
            #response = client.read_coils(address=320, count=8, slave=0)
            #value = response.registers
            #q_read_msg.put([tagname_floodgate_down, value])
            #await asyncio.sleep(0.2)

            print("Read floodgate position")
            response = client.read_holding_registers(address=14000, count=1, slave=0)
            value = response.registers[0]
            value = (2.5/60)*value
            print(value)
            q_read_msg.put([tagname_floodgate_pos, value])
            await asyncio.sleep(1)
        except Exception as e:
            print("Read MODBUS fail!")
            print(e)
            await asyncio.sleep(5)



async def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-j", "--json", help="JSON file for the configuration", default='config.json')

    # Read arguments from command line
    args = parser.parse_args()

    # URL for video or camera source
    f = open(args.json)
    data = json.load(f)
    username = data['redis_user']
    password = data['redis_pass']
    f.close()

    # shared queue
    q_write_msg = queue.Queue()
    q_read_msg = queue.Queue()

    redis = aioredis.from_url('redis://localhost', username=username, password=password)
    pubsub = redis.pubsub()
    await pubsub.subscribe('settag:'+tagname_floodgate_up)
    await pubsub.subscribe('settag:'+tagname_floodgate_down)
    asyncio.create_task(redis_sub(pubsub, q_write_msg))
    asyncio.create_task(modbus_worker(q_write_msg, q_read_msg))
    asyncio.create_task(redis_pub(redis, q_read_msg))

    while True:
        print("waiting for message...")
        await asyncio.sleep(5)



if __name__ == '__main__':
    asyncio.run(main())
