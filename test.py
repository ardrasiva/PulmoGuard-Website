from pymongo import MongoClient

uri = "mongodb+srv://ardrasiva123:ardmongo1612@cluster0.wqlfkjh.mongodb.net/"

client = MongoClient(uri)
db = client["pulmoguard"]
print(db.list_collection_names())
