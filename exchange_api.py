import pyupbit
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os
import logging

# 로깅 설정
logging.basicConfig(filename='trade_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 환경 변수 가져오기
ACCESS_KEY = os.getenv("ACCESS_KEY", "your-access-key")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
TRADE_MODE = os.getenv("TRADE_MODE", "mock")  # "mock" 또는 "real"
TARGET_PROFIT_RATE = 0.02  # 목표 수익률 2%

upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 모델 로드 및 학습 진행 상태 저장
model_file = "model.h5"
sequence_length = 50
model = None


# 데이터 수집 및 모델 준비
def prepare_model():
    global model

    ticker = "KRW-DOGE"
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=1000)  # 더 많은 데이터 수집

    # 데이터 전처리
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']])
    x_data, y_data = [], []

    # LSTM 모델 입력 생성
    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i - sequence_length:i])
        y_data.append(scaled_data[i, 0])
    x_data, y_data = np.array(x_data), np.array(y_data)

    # 모델 생성 또는 로드
    if os.path.exists(model_file):
        model = load_model(model_file)
        logging.info("Saved model loaded.")
    else:
        # 모델 생성 및 학습
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_data, y_data, epochs=10, batch_size=32, verbose=1)
        model.save(model_file)
        logging.info("Model saved.")

    return scaler


# @tf.function으로 예측 함수 정의
@tf.function
def predict_with_model(model, input_data):
    return model(input_data)


# 데이터 수집 및 예측
def get_data_and_predict(scaler):
    ticker = "KRW-DOGE"
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)

    # 데이터 전처리
    scaled_data = scaler.transform(df[['close']])

    # 예측
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    last_sequence_tensor = tf.convert_to_tensor(last_sequence, dtype=tf.float32)  # 텐서로 변환
    predicted_price = predict_with_model(model, last_sequence_tensor)
    predicted_price = scaler.inverse_transform([[predicted_price.numpy()[0][0]]])[0][0]

    logging.info(f"Predicted price: {predicted_price}")
    return predicted_price


# 거래 로직
def trade_logic(scaler):
    try:
        predicted_price = get_data_and_predict(scaler)
        current_price = pyupbit.get_current_price("KRW-DOGE")
        balance = upbit.get_balance("KRW") if TRADE_MODE == "real" else 100000  # 모의 거래 초기 자본

        logging.info(f"Predicted: {predicted_price}, Current: {current_price}, Balance: {balance}")

        if balance > 5000 and predicted_price > current_price:
            buy_amount = 5000
            if TRADE_MODE == "real":
                upbit.buy_market_order("KRW-DOGE", buy_amount)
                logging.info(f"Real Trade - Buy {buy_amount} KRW")
            else:
                balance -= buy_amount
                logging.info(f"Mock Trade - Buy {buy_amount} KRW, Remaining Balance: {balance}")

        # 매도 로직 (목표 수익률 도달 시)
        doge_balance = upbit.get_balance("DOGE") if TRADE_MODE == "real" else 50
        avg_buy_price = current_price * (1 - TARGET_PROFIT_RATE)

        if doge_balance > 0 and current_price >= avg_buy_price * (1 + TARGET_PROFIT_RATE):
            if TRADE_MODE == "real":
                upbit.sell_market_order("KRW-DOGE", doge_balance)
                logging.info(f"Real Trade - Sell DOGE")
            else:
                balance += doge_balance * current_price
                logging.info(f"Mock Trade - Sell DOGE, Updated Balance: {balance}")
    except Exception as e:
        logging.error(f"오류 발생: {e}")

