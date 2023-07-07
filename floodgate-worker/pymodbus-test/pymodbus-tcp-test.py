import time
from pymodbus.client import ModbusTcpClient



def test_coil(mbclient):
    print("test write Coils")
    try:
        mbclient.write_coils(address=160, values=[1, 1, 0, 0, 0, 0, 0, 1], slave=0)
        time.sleep(1)
    except Exception as e:
        print(e)

    print("test read Coils")
    try:
        response = mbclient.read_coils(address=160, count=8, slave=0)
        time.sleep(1)
        print("response={}".format(response.bits))
    except Exception as e:
        print(e)


def test_hold_register(mbclient):
    print("test read HR")
    try:
        time.sleep(1)
        response = mbclient.read_holding_registers(address=14000, count=1, slave=0)
        print(response.registers[0])
    except Exception as e:
        print(e)



def main():
    print("pymodbus TCP testing...")
    time.sleep(1)

    client = ModbusTcpClient('192.168.0.50')
    client.connect()
    #test_coil(client)
    while True:
        test_hold_register(client)
        time.sleep(0.2)
    print("test end")




if __name__ == '__main__':
    main()
