import uuid

def generate_key():
    unique_id = uuid.uuid4().hex
    return f"sk-{unique_id}"

key = generate_key()
print(key)