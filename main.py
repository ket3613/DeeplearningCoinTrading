from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from exchange_api import trade_logic, prepare_model
import os
import logging

# 로깅 설정
logging.basicConfig(filename='server_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

# 모델 준비 및 스케일러 생성
scaler = prepare_model()

# 백그라운드 스케줄러 설정
scheduler = BackgroundScheduler()

# 거래 주기 설정 (5분마다 실행)
scheduler.add_job(trade_logic, 'interval', minutes=5, args=[scaler])
scheduler.start()
logging.info("Scheduler started.")

# 헬스 체크 엔드포인트
@app.get("/health")
async def health():
    return {"status": "ok"}

# 서버 중지 시 스케줄러 중단
@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    logging.info("Scheduler shutdown.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("SERVER_HOST", "0.0.0.0"), port=int(os.getenv("SERVER_PORT", 8000)))
