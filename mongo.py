import os
from typing import Dict, Optional, Any, List, AsyncGenerator
from pathlib import Path
import logging

from pydantic.dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCursor
from pymongo.server_api import ServerApi
from pymongo.results import InsertOneResult, InsertManyResult, DeleteResult, UpdateResult



class MongoHandler:
    def __init__(self, uri: str, cert_path: str) -> None:
        self.client: AsyncIOMotorClient = self._init_client(
            uri=uri,
            cert_path=cert_path,
        )


    def _init_client(self, uri: str, cert_path: str):
        try:
            client = AsyncIOMotorClient(
                uri,
                tls=True,
                tlsCertificateKeyFile=cert_path,
                server_api=ServerApi('1'),
            )
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise ValueError('Could not connect to MongoDB')
        
        return client


    async def test_database_connection(self):
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logging.error(f"Database connection test failed: {e}")
            return False


    def close_connection(self) -> None:
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")


    async def count_documents(self, db_name: str, collection_name: str, filter: Dict = {}) -> int:
        collection =  self.get_collection(db_name=db_name, collection_name=collection_name)
        return await collection.count_documents(filter=filter)


    async def insert(self, entry: dict, db_name: str = 'data', collection_name: str = 'url_map') -> InsertOneResult:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        result = await collection.insert_one(entry)

        return result


    async def insert_many(self, entries: List[dict], db_name: str = 'data', collection_name: str = 'url_map') -> InsertManyResult:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        result = await collection.insert_many(entries)
        return result


    async def get_document(self, db_name: str = 'data', collection_name: str = 'url_map', filter: Optional[Dict] = None, sort: Optional[list] = None) -> Optional[Dict]:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        filter = filter if filter is not None else {}
        if sort:
            return await collection.find_one(filter=filter, sort=sort)
        return await collection.find_one(filter=filter)
    

    async def get_documents(self, db_name: str, collection_name: str, query: dict, length: int = None) -> AsyncGenerator[List[Dict[str, Any]], None]:
        collection = self.get_collection(db_name, collection_name)
        cursor = collection.find(query)

        while True:
            documents = await cursor.to_list(length=length)

            if not documents:
                break

            yield documents
            
        await cursor.close()


    async def delete_document(self, db_name: str = 'data', collection_name: str = 'url_map', filter: Optional[Dict] = {}) -> DeleteResult:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        return await collection.delete_one(filter=filter)


    async def delete_documents(self, db_name: str = 'data', collection_name: str = 'url_map', filter: Optional[Dict] = {}) -> DeleteResult:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        return await collection.delete_many(filter=filter)


    async def update_document(self, entry: Dict, db_name: str = 'data', collection_name: str = 'url_map', query: Optional[Dict] = None) -> UpdateResult:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        return await collection.update_one(query, {"$set": entry}, upsert=True)


    async def document_exists(self, db_name: str = 'data', collection_name: str = 'url_map', filter: Optional[Dict] = None) -> int:
        collection = self.get_collection(db_name=db_name, collection_name=collection_name)
        return await collection.count_documents(filter=filter)


    def get_collection(self, db_name: str, collection_name: str):
        return self.client[db_name][collection_name]


    async def create_document(self, **kwargs) -> Dict:
        return {key:value for key, value in kwargs.items()}


    async def cleanup(self, db_name: str, collection_name: str) -> DeleteResult:
        return await self.delete_documents(db_name=db_name, collection_name=collection_name)