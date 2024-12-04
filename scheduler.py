from apscheduler.schedulers.blocking import BlockingScheduler
from exchange_api import trade_logic

scheduler = BlockingScheduler()

# 5분 간격으로 거래 로직 실행
scheduler.add_job(trade_logic, 'interval', minutes=5)

if __name__ == "__main__":
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
