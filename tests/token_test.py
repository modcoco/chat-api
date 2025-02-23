from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class User(BaseModel):
    userId: str


def create_access_token(
    data: dict,
    expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
):
    to_encode = data.copy()
    expire = datetime.now() + expires_delta
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None


def main():
    # 创建一个用户对象
    user = User(userId="12345")

    # 创建JWT Token
    token_data = {"userId": user.userId}
    token = create_access_token(token_data)
    print("Generated Token:")
    print(token)

    # # 验证JWT Token
    # decoded_data = verify_token(token)
    # if decoded_data:
    #     print("\nDecoded Token Data:")
    #     print(decoded_data)
    # else:
    #     print("\nToken is invalid or expired.")


if __name__ == "__main__":
    main()
