from apscheduler.schedulers.background import BackgroundScheduler

class Scheduler:
    def __init__(self):
        self._sched = BackgroundScheduler()
        self._sched.start()

    def schedule_daily(self, hour: int, minute: int, func, job_id: str, args=None):
        self._sched.add_job(
            func,
            'cron',
            hour=hour,
            minute=minute,
            args=args or [],
            id=job_id,
            replace_existing=True
        )

    def remove(self, job_id: str):
        try:
            self._sched.remove_job(job_id)
        except:
            pass

    def shutdown(self):
        self._sched.shutdown()
