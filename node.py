from typing import Optional
import struct
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from graph import Graph


class Node(BaseModel):
    id: int
    name: str
    data: bytearray = Field(default=bytearray(10))
    subgraph: Optional['Graph'] = None

    def update_engagement(self) -> None:
        self.set_interest_frequency()
        self.set_last_engagement()

    def update_engagement_score(self):
        recency = (datetime.now(timezone.utc) - self.get_last_engagement()).total_seconds()
        self.engagement_score = (self.get_interest_frequency() * self.get_eigenvector_centrality()) / (recency + 1)

    def set_interest_frequency(self) -> None:
        new_value = 1 + self.get_interest_frequency()
        struct.pack_into('H', self.data, 0, new_value)

    def get_interest_frequency(self) -> int:
        return struct.unpack_from('H', self.data, 0)[0]

    def set_eigenvector_centrality(self, value: float) -> None:
        struct.pack_into('H', self.data, 2, int(value * 65535))

    def get_eigenvector_centrality(self) -> float:
        return struct.unpack_from('H', self.data, 2)[0] / 65535.0
    
    def set_engagement_score(self, value: float) -> None:
        struct.pack_into('H', self.data, 4, int(value * 65535))

    def get_engagement_score(self) -> float:
        return struct.unpack_from('H', self.data, 4)[0] / 65535.0
    
    def set_last_engagement(self) -> None:
        value = int(datetime.now(timezone.utc).timestamp())
        struct.pack_into('I', self.data, 6, value)

    def get_last_engagement(self) -> datetime:
        return datetime.fromtimestamp(struct.unpack_from('I', self.data, 6)[0])
    
    def set_subgraph(self, subgraph: 'Graph') -> None:
        self.subgraph = subgraph

    def to_feature_vector(self):
        return [
            self.get_interest_frequency(),
            self.get_eigenvector_centrality(),
            self.get_engagement_score(),
            (datetime.now(timezone.utc) - self.get_last_engagement()).total_seconds()
        ]

    def to_dict(self):
        return {
            'i': self.id,
            'd': self.data,
        }