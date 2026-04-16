from nautilus_trader.config import LiveDataClientConfig, LiveExecClientConfig


class AlpacaDataClientConfig(LiveDataClientConfig, frozen=True, kw_only=True):
    """Configuration for Alpaca data client."""

    api_key: str
    secret_key: str
    paper: bool = True

    @property
    def base_url(self) -> str:
        return "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

    @property
    def data_ws_url(self) -> str:
        return "wss://stream.data.alpaca.markets/v2/iex"


class AlpacaExecClientConfig(LiveExecClientConfig, frozen=True, kw_only=True):
    """Configuration for Alpaca execution client."""

    api_key: str
    secret_key: str
    paper: bool = True

    @property
    def base_url(self) -> str:
        return "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
