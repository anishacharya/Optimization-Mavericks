def get_sampling_scheduler(schedule: str = 'step',
                           step_size: int = 100,
                           beta: float = 1):
    if schedule == 'step':
        return StepSamplingSchedule(step_size=step_size, beta=beta)
    else:
        raise NotImplementedError


class SamplingScheduler:
    def __init__(self, k0: int = 1):
        self.k0 = k0
        self._step_count = 0

    def step(self) -> float:
        raise NotImplementedError("This method needs to be implemented for each Scheduling Algorithm")


class StepSamplingSchedule(SamplingScheduler):
    def __init__(self,
                 step_size: int,
                 beta: float = 0.1):
        SamplingScheduler.__init__(self)
        self.step_size = step_size
        self.beta = beta

    def step(self) -> float:
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            self.k0 = self.k0 * self.beta
            if self.k0 == 0:
                self.k0 += 1
            print('updating sample fraction at step {} to {}'.format(self._step_count, self.k0))
        return self.k0


class MultiStepSamplingSchedule(SamplingScheduler):
    def __init__(self, milestones, total_steps: int, beta: float = 0.1):
        SamplingScheduler.__init__(self)
        self.milestones = list(sorted(map(lambda x: int(total_steps * x), milestones)))

    def step(self):
        pass


# Test Script
if __name__ == '__main__':
    sampling_scheduler = StepSamplingSchedule(step_size=25,
                                              beta=0.1)
    for j in range(100):
        sampling_scheduler.step()
