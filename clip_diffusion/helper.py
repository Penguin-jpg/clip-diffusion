import anvil.server


class Helper:
    """
    幫忙包裝、簡化anvil的一些function
    """

    def __init__(self, uplink_key):
        self.uplink_key = uplink_key
        self.connect_to_anvil()

    def connect_to_anvil(self):
        """
        連線到anvil
        """

        anvil.server.connect(self.uplink_key)

    def start_server():
        """
        server開始等到呼叫
        """

        anvil.server.wait_forever()
